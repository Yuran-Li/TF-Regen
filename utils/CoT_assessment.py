import utils.prompts as prompts
import utils.models as models
import json
import random
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import joblib
from tqdm import tqdm
import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
# Fix OpenBLAS warning by setting thread limits before using SentenceTransformer
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
from .check import check_answer_gsm_hard, check_answer_strategyqa, check_answer_gsm8k, check_answer_math,check_answer_coin, check_answer_letter, check_answer_legal
from .check import extract_answer
from .quality import get_prediction, split_assessment
# unified dataset configuration
from .CoT_generation import DATASET_CONFIG

def shannon_entropy(frequencies):
    total_responses = sum(frequencies)
    if total_responses == 0:
        return 0  
    entropy = 0
    for freq in frequencies:
        if freq > 0:
            p_i = freq / total_responses  
            entropy -= p_i * math.log(p_i, 2)  
    return entropy

ENTROPY_THRESHOLD = 0.1

class CoTAssessment:
    def __init__(self, model_name, model_type, dataset_name, method, eg_path, temperature=0, top_p=1.0, max_tokens=1024, seed=42):
        self.model_name = model_name
        self.model_type = model_type
        self.dataset_name = dataset_name
        self.method = method
        self.eg_path = eg_path
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.seed = seed
        
        # set random seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Only load eg_data if eg_path is provided
        if self.eg_path is not None:
            self.eg_data = json.load(open(self.eg_path, "r"))
        else:
            self.eg_data = None
            
        if model_type == "api":
            self.api = models.get_chat_api_from_model(self.model_name)
        elif model_type == "local":
            # LOCAL_BACKEND: "vllm" (default) or "hf"
            local_backend = os.environ.get("LOCAL_BACKEND", "vllm")
            self.api = models.get_local_api(self.model_name, backend=local_backend)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
        
        # Load local model for Referi score calculation (if method is Referi_score and model_type is local)
        self.referi_model = None
        self.referi_tokenizer = None
        if self.method == "Referi_score" and self.model_type == "local":
            self._load_referi_model()
        
        # Pre-load sentence transformer model and embeddings for similarity method
        if (self.method == "contrastive_guided" or self.method == "Referi_score") and self.eg_data is not None:
            self.sentence_model = SentenceTransformer("all-mpnet-base-v2", device="cpu")
            ## all items
            self.all_items = self.eg_data
            ## good items
            self.good_items = [item for item in self.all_items if isinstance(item, dict) and item.get("assessment") == "good"]
            ## filtered items
            self.filtered_items = self.good_items.copy()
            self.error_items = []
            for item in self.all_items:
                has_reflection = False
                if isinstance(item["assessment"], dict):
                    analysis_1, corrected_1 = split_assessment(item["assessment"].get("rectify"))
                    analysis_2 = item["assessment"].get("reflect")
                    corrected_2 = item["assessment"].get("reflective_rectify")
                    principles = item["assessment"].get("principles")
                    has_reflection = True
                if has_reflection:
                    # use unified config for check functions
                    config = DATASET_CONFIG.get(self.dataset_name)
                    if config:
                        check_func = config["check_func"]
                        is_correct = (check_func(corrected_1, item[config["answer_field"]]))
                        if is_correct:
                            self.filtered_items.append(item)
                            self.error_items.append(item)
            print(f"length of filtered_items: {len(self.filtered_items)}")
            print(f"length of error_items: {len(self.error_items)}")
            print(f"length of good_items: {len(self.good_items)}")
            ## embedding filtered items
            ## use unified config for field mapping
            field_mapping = {dataset: config["question_field"] for dataset, config in DATASET_CONFIG.items()}
            if self.dataset_name not in field_mapping:
                raise ValueError(f"Unsupported dataset: {self.dataset_name}")
            '''
            embedding method 1:
            question based similarity
            only select examples with similar questions embeddings
            '''
            self.sentence_model_mpnet = SentenceTransformer("all-mpnet-base-v2", device="cpu")
            ## use OpenAI embedding model
            self.good_eg_Q_embeddings = self.sentence_model_mpnet.encode(
                [item[field_mapping[self.dataset_name]] for item in self.good_items],
                normalize_embeddings=True
            )
            self.error_eg_Q_embeddings = self.sentence_model_mpnet.encode(
                [item[field_mapping[self.dataset_name]] for item in self.error_items],
                normalize_embeddings=True
            )
            '''
            embedding method 2:
            example based similarity
            select examples with helpful questions, reasoning and answer
            '''
            self.sentence_model_contriever = SentenceTransformer("facebook/contriever", device="cpu") #facebook/contriever-msmarco
            self.good_eg_A_embeddings = self.sentence_model_contriever.encode(
                [
                    f"Answer:{item.get('eg')}"
                    for item in self.good_items
                ],
                normalize_embeddings=True,
            )
            self.error_eg_A_embeddings = self.sentence_model_contriever.encode(
                [
                    f"Answer:{split_assessment(item['assessment'].get('rectify'))[1]}\n"
                    for item in self.error_items
                ],
                normalize_embeddings=True,
            )


    ## assess response in multiple runs
    def entropy_based_assessment(self, data):
        predictions = data["CoT"]
        # calculate entropy, only when two or more CoT included
        if not isinstance(predictions, list) or len(predictions) <= 1:
            return "already assessed"
        statistics = {}
        cleaned_predictions = []
        for prediction in predictions:
            p = extract_answer(prediction)
            statistics[p] = statistics.get(p, 0) + 1
            if p is None:
                continue
            cleaned_predictions.append(p)
        
        if not cleaned_predictions:
             return "incorrect"
        ## calcukate the Shannon entropy of p
        entropy = shannon_entropy(list(statistics.values()))
        if entropy > ENTROPY_THRESHOLD:
            assessment = "incorrect"
        else:
            assessment = "correct"
        return assessment

    ## assess response in a single run
    def label_based_assessment(self, data):
        config = DATASET_CONFIG.get(self.dataset_name)
        if not config:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
        prediction = data["CoT"]
        label = data[config['answer_field']]
        check_func = config["check_func"]
        is_correct = check_func(prediction, label)
        if is_correct:
            assessment = "correct"
        else:
            assessment = "incorrect"
        return assessment

    ## assess response in a single run
    async def contrastive_guided_assessment(self, data):
        if extract_answer(data["CoT"]) is None:
            return "incorrect"
        selected_items = []
        if self.eg_data is None:
            raise ValueError("Example data is required for 'random' method. Please provide eg_path parameter.")
        # set seed for random sampling to ensure reproducibility
        random.seed(self.seed)

        ## similarity select items
        ## use unified config for field mapping
        field_mapping = {dataset: config["question_field"] for dataset, config in DATASET_CONFIG.items()}
        question_embedding_mpnet = self.sentence_model_mpnet.encode(data[field_mapping[self.dataset_name]], normalize_embeddings=True).reshape(1, -1)     
        query_text = f"Question: {data[field_mapping[self.dataset_name]]}"
        question_embedding_contriver = self.sentence_model_contriever.encode(query_text, normalize_embeddings=True).reshape(1, -1)
        answer_embedding_contriver = self.sentence_model_contriever.encode(data["CoT"], normalize_embeddings=True).reshape(1, -1)
        
        ## Two-stage retrieval: 
        ## Configurable parameters
        num_candidates = 10  # Stage 1 
        num_good_examples = 1  # Stage 2 
        num_error_examples = 1  # Stage 2 
        
        ## Stage 1: Contriver - rough selection (top K) based on cross-modal Q-A similarity
        # similarities = cosine_similarity(question_embedding_contriver, self.good_eg_A_embeddings)
        similarities = cosine_similarity(answer_embedding_contriver, self.good_eg_A_embeddings)
        stage1_good_indices = np.argsort(similarities[0])[-num_candidates:][::-1]
        
        # similarities = cosine_similarity(question_embedding_contriver, self.error_eg_A_embeddings)
        similarities = cosine_similarity(answer_embedding_contriver, self.error_eg_A_embeddings)
        stage1_error_indices = np.argsort(similarities[0])[-num_candidates:][::-1]
        
        ## Stage 2: MPNet - fine selection (top N) based on Q-Q similarity (reuse precomputed embeddings)
        candidates_good_Q_embeddings = self.good_eg_Q_embeddings[stage1_good_indices]
        similarities = cosine_similarity(question_embedding_mpnet, candidates_good_Q_embeddings)
        stage2_good_relative_indices = np.argsort(similarities[0])[-num_good_examples:][::-1]
        final_good_indices = stage1_good_indices[stage2_good_relative_indices]
        
        candidates_error_Q_embeddings = self.error_eg_Q_embeddings[stage1_error_indices]
        similarities = cosine_similarity(question_embedding_mpnet, candidates_error_Q_embeddings)
        stage2_error_relative_indices = np.argsort(similarities[0])[-num_error_examples:][::-1]
        final_error_indices = stage1_error_indices[stage2_error_relative_indices]
        
        selected_items = [self.good_items[i] for i in final_good_indices] + [self.error_items[i] for i in final_error_indices]
        config = DATASET_CONFIG.get(self.dataset_name)
        if not config:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
        
        # generate examples
        examples_list = []
        for item in selected_items:
            has_reflection = False
            if isinstance(item['eg'], str):
                prediction = item['eg']
                is_correct = 0
            else:
                prediction, is_correct = get_prediction(item['eg'], item[config['answer_field']], self.dataset_name)
            if isinstance(item["assessment"], dict):
                analysis_1, corrected_1 = split_assessment(item["assessment"].get("rectify"))
                analysis_2 = item["assessment"].get("reflect")
                corrected_2 = item["assessment"].get("reflective_rectify")
                principle = item["assessment"].get("principles")
                has_reflection = True
            
            if item["assessment"] == "bad":
                continue
            elif item["assessment"] == "good":
                example = f"Question: {item[config['question_field']]}\nAnswer:{prediction}\nAnalysis: The answer is correct.\nAssessment:correct\n\n"
                # pass
            elif has_reflection:
                example = f"Question: {item[config['question_field']]}\nAnswer:{prediction}\nAnalysis: {analysis_1}\nAssessment:incorrect\n\n"
                # example = f"{principle}\n"

            examples_list.append(example)
        examples = "\n\n".join(examples_list)
        prompt = prompts.render_template(
            config["template_verify"],
            question=data[config["question_field"]], 
            answer=data["CoT"],
            examples=examples
        )
        # print(f"prompt: {prompt}")
        assessment = await self.api.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            top_p=1.0,
            max_tokens=1024,
        )
        # print(f"assessment: {assessment}")
        return assessment


    async def self_guided_assessment(self, data):
        config = DATASET_CONFIG.get(self.dataset_name)
        if not config:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
        prompt = prompts.render_template(
            "zero_shot_assessment_prompt",
            question=data[config["question_field"]], 
            answer=data["CoT"]
        )

        assessment = await self.api.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            top_p=1.0,
            max_tokens=1024,
        )
        return assessment
    
    def _load_referi_model(self):
        """ Referi """
        model_path = self._resolve_model_path(self.model_name)
        self.referi_tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.referi_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        ).eval()
    
    def _resolve_model_path(self, model_name_or_path):
        """ HuggingFace """
        if os.path.exists(model_name_or_path) and os.path.isdir(model_name_or_path):
            return model_name_or_path
        
        model_mapping = {
            "Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
            "Llama-3.1-70B-Instruct": "meta-llama/Llama-3.1-70B-Instruct",
            "Llama-3.3-70B-Instruct": "meta-llama/Llama-3.3-70B-Instruct",
        }
        
        if model_name_or_path in model_mapping:
            return model_mapping[model_name_or_path]
        
        if "/" in model_name_or_path:
            return model_name_or_path
        
        return f"meta-llama/{model_name_or_path}"
    
    def _calc_loss(self, prompt, target_answer):
        input_text = prompt + target_answer
        inputs = self.referi_tokenizer(input_text, return_tensors="pt")
        
        if hasattr(self.referi_model, 'device'):
            device = self.referi_model.device
        elif hasattr(self.referi_model, 'hf_device_map'):
            device = next(iter(self.referi_model.hf_device_map.values()))['device']
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        labels = inputs['input_ids'].clone()
        prompt_len = len(self.referi_tokenizer(prompt, return_tensors="pt")['input_ids'][0])
        labels[:, :prompt_len] = -100  # Ignore prompt
        inputs['labels'] = labels

        with torch.no_grad():
            outputs = self.referi_model(**inputs)
        
        return outputs.loss.item()

    async def Referi_score_assessment(self, data):
        if extract_answer(data["CoT"]) is None:
            return "incorrect"
        selected_items = []
        if self.eg_data is None:
            raise ValueError("Example data is required for 'random' method. Please provide eg_path parameter.")
        # set seed for random sampling to ensure reproducibility
        random.seed(self.seed)

        ## similarity select items
        ## use unified config for field mapping
        field_mapping = {dataset: config["question_field"] for dataset, config in DATASET_CONFIG.items()}
        question_embedding_mpnet = self.sentence_model_mpnet.encode(data[field_mapping[self.dataset_name]], normalize_embeddings=True).reshape(1, -1)     
        query_text = f"Question: {data[field_mapping[self.dataset_name]]}"
        question_embedding_contriver = self.sentence_model_contriever.encode(query_text, normalize_embeddings=True).reshape(1, -1)
        answer_embedding_contriver = self.sentence_model_contriever.encode(data["CoT"], normalize_embeddings=True).reshape(1, -1)
        
        ## Two-stage retrieval: 
        ## Configurable parameters
        num_candidates = 10  # Stage 1 
        num_good_examples = 1  # Stage 2
        num_error_examples = 1  # Stage 2
        
        ## Stage 1: Contriver - rough selection (top K) based on cross-modal Q-A similarity
        # similarities = cosine_similarity(question_embedding_contriver, self.good_eg_A_embeddings)
        similarities = cosine_similarity(answer_embedding_contriver, self.good_eg_A_embeddings)
        stage1_good_indices = np.argsort(similarities[0])[-num_candidates:][::-1]
        
        # similarities = cosine_similarity(question_embedding_contriver, self.error_eg_A_embeddings)
        similarities = cosine_similarity(answer_embedding_contriver, self.error_eg_A_embeddings)
        stage1_error_indices = np.argsort(similarities[0])[-num_candidates:][::-1]
        
        ## Stage 2: MPNet - fine selection (top N) based on Q-Q similarity (reuse precomputed embeddings)
        candidates_good_Q_embeddings = self.good_eg_Q_embeddings[stage1_good_indices]
        similarities = cosine_similarity(question_embedding_mpnet, candidates_good_Q_embeddings)
        stage2_good_relative_indices = np.argsort(similarities[0])[-num_good_examples:][::-1]
        final_good_indices = stage1_good_indices[stage2_good_relative_indices]
        
        candidates_error_Q_embeddings = self.error_eg_Q_embeddings[stage1_error_indices]
        similarities = cosine_similarity(question_embedding_mpnet, candidates_error_Q_embeddings)
        stage2_error_relative_indices = np.argsort(similarities[0])[-num_error_examples:][::-1]
        final_error_indices = stage1_error_indices[stage2_error_relative_indices]
        
        examples_list = [self.good_items[i] for i in final_good_indices]# + [self.error_items[i] for i in final_error_indices]
        config = DATASET_CONFIG.get(self.dataset_name)
        if not config:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

        few_shots = []
        for item in examples_list:
            question = item[config['question_field']]
            answer = item['eg'][0]
            few_shots.append((question, answer))
        
        if not few_shots:
            return "incorrect"  

        question = data[config["question_field"]]
        model_output = data["CoT"]
        
        # Referi Score
        score = self._score_single_output(question, model_output, few_shots, config)
        
        threshold = 0.0  # if score > 0 (mean reduce Loss), Correct
        assessment = "correct" if score > threshold else "incorrect"
        return assessment
    
    def _score_single_output(self, question, model_output, few_shots, config):
        """
        对单个生成结果打分 (完全对齐 Referi 逻辑)
        Score = Average( Loss_No_Replace - Loss_Replace ) over all reference examples
        """
        if self.referi_model is None or self.referi_tokenizer is None:
            raise ValueError("Referi model not loaded. Please ensure model_type is 'local' and method is 'Referi_score'")
        
        total_score = 0
        valid_count = 0
        
        # system prompt
        system_prompt = "Answer the given question step by step. And put the final answer after '####' in the end.\n"
        
        # iterate over each few-shot sample as Target
        for i, (target_q, target_a) in enumerate(few_shots):
            # construct Baseline Prompt (No Replace)
            context_str = system_prompt
            for j, (q, a) in enumerate(few_shots):
                if i == j: continue  # skip the current sample as Target
                context_str += f"Q: {q}\nA: {a}\n\n"
            
            baseline_prompt = f"{context_str}Q: {target_q}\nA:"
            
            # construct Referi Prompt (Replace)
            referi_prompt = f"{context_str}Q: {question}\nA: {model_output}\n\nQ: {target_q}\nA:"
            
            # calculate Loss
            try:
                loss_baseline = self._calc_loss(baseline_prompt, target_a)
                loss_referi = self._calc_loss(referi_prompt, target_a)
                
                # Score > 0 means that adding model_output reduces the loss of target_a (confidence提升)
                score = loss_baseline - loss_referi
                
                total_score += score
                valid_count += 1
                
            except Exception as e:
                print(f"Error calculating score for shot {i}: {e}")
                continue

        # return average score
        if valid_count == 0:
            return -999.0
            
        return total_score / valid_count

    async def CoT_WP_assessment(self, data):
        """
        CoT-WP (Chain-of-Thought with Wrong Prediction) assessment.
        Based on the difference between the Top-1 and Top-2 token probabilities during generation to determine confidence.
        """
        # check if there is logprobs data
        if 'logprobs' not in data or data['logprobs'] is None:
            # if there is no logprobs, trigger regeneration to get logprobs
            logprobs = await self._regenerate_with_logprobs(data)
            if logprobs is None or not logprobs:
                return "incorrect"
        else:
            logprobs = data['logprobs']
            if not logprobs:
                return "incorrect"
        
        # get the generated text (if any, for locating "####")
        generated_text = data.get("CoT", None)
        
        # calculate CoT-WP score (only evaluate the part after "####")
        score = self._calculate_cot_wp_score(logprobs, generated_text)
        
        # determine correct or incorrect based on the threshold
        # the threshold needs to be adjusted based on实际情况，通常 score 越高表示置信度越高
        threshold = 0.1  # adjustable threshold
        assessment = "correct" if score > threshold else "incorrect"
        return assessment
    
    async def _regenerate_with_logprobs(self, data):
        """
        Regenerate CoT and get logprobs data.
        
        Returns:
            list: logprobs list, format as [{'token': '...', 'logprob': -0.1, 'top_logprobs': {...}}, ...]
                  if generation fails or API is not supported, return None
        """
        config = DATASET_CONFIG.get(self.dataset_name)
        if not config:
            return None
        
        # construct prompt (using zero-shot template)
        question = data[config["question_field"]]
        prompt = prompts.render_template(config.get("zero_shot_template"), question=question)
        
        # call API and get logprobs
        try:
            # check API type
            if isinstance(self.api, models.OpenAIAPI):
                # OpenAI API supports logprobs
                response = await self.api.client.chat.completions.create(
                    model=self.api.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens,
                        logprobs=True,  # enable logprobs
                    top_logprobs=5,  # return top 5 logprobs
                )
                
                # extract logprobs
                if response.choices and response.choices[0].logprobs:
                    logprobs = []
                    for content_token in response.choices[0].logprobs.content:
                        token_data = {
                            'token': content_token.token,
                            'logprob': content_token.logprob,
                            'top_logprobs': {item.token: item.logprob for item in content_token.top_logprobs}
                        }
                        logprobs.append(token_data)
                    return logprobs
            elif isinstance(self.api, models.VLLMAPI):
                # vLLM API also supports logprobs
                # use backoff retry mechanism to handle connection errors
                import backoff
                import openai
                
                @backoff.on_exception(backoff.fibo, (openai.OpenAIError, ConnectionError), max_tries=3, max_value=10)
                async def _call_with_retry():
                    response = await self.api.client.chat.completions.create(
                        model=self.api.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.temperature,
                        top_p=self.top_p,
                        max_tokens=self.max_tokens,
                        logprobs=True,
                        top_logprobs=5,
                    )
                    return response
                
                try:
                    response = await _call_with_retry()
                    
                    if response.choices and response.choices[0].logprobs:
                        logprobs = []
                        for content_token in response.choices[0].logprobs.content:
                            token_data = {
                                'token': content_token.token,
                                'logprob': content_token.logprob,
                                'top_logprobs': {item.token: item.logprob for item in content_token.top_logprobs}
                            }
                            logprobs.append(token_data)
                        return logprobs
                except Exception as e:
                    # silently handle errors, do not print warnings (because they are already handled in the retry)
                    return None
            else:
                # for other API types, try generic approach
                if hasattr(self.api, 'client'):
                    try:
                        response = await self.api.client.chat.completions.create(
                            model=self.api.model,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=self.temperature,
                            top_p=self.top_p,
                            max_tokens=self.max_tokens,
                            logprobs=True,
                            top_logprobs=5,
                        )
                        
                        if response.choices and response.choices[0].logprobs:
                            logprobs = []
                            for content_token in response.choices[0].logprobs.content:
                                token_data = {
                                    'token': content_token.token,
                                    'logprob': content_token.logprob,
                                    'top_logprobs': {item.token: item.logprob for item in content_token.top_logprobs}
                                }
                                logprobs.append(token_data)
                            return logprobs
                    except Exception as e:
                        # silently handle errors, do not print warnings
                        return None
        except Exception as e:
            # only print errors when final failure occurs
            return None
        
        return None
    
    def _calculate_cot_wp_score(self, logprobs, generated_text=None):
        """
        Calculate CoT-WP score: based on the average difference between the Top-1 and Top-2 token probabilities.
        Only evaluate the part after "####" (answer part).
        
        Args:
            logprobs: list of dicts, each dict contains 'top_logprobs' for a token.
                     Format: [{'token': '2', 'logprob': -0.1, 'top_logprobs': {'2': -0.1, '3': -2.5, ...}}, ...]
            generated_text: optional, the complete generated text, for locating the position of "####"
        
        Returns:
            float: average probability difference, the higher the value, the higher the confidence
        """
        if not logprobs:
            return 0.0
        
        # find the part after "####"
        answer_start_idx = self._find_answer_start_index(logprobs, generated_text)
        
        # only use logprobs after "####"
        if answer_start_idx is not None and answer_start_idx < len(logprobs):
            target_tokens = logprobs[answer_start_idx:]
        else:
            # if "####" is not found, return 0.0 (will be judged as incorrect)
            return 0.0
        
        if not target_tokens:
            return 0.0
        
        diffs = []
        
        for token_data in target_tokens:
            # get logprobs of Top-2
            # OpenAI API returns logprob, need to convert back to probability using exp
            if 'top_logprobs' not in token_data or not token_data['top_logprobs']:
                continue
                
            top_probs = sorted([np.exp(lp) for lp in token_data['top_logprobs'].values()], reverse=True)
            
            if len(top_probs) >= 2:
                p_top1 = top_probs[0]
                p_top2 = top_probs[1]
                diffs.append(p_top1 - p_top2)
        
        if not diffs:
            return 0.0
        
        # calculate average difference as the final score
        return np.mean(diffs)
    
    def _find_answer_start_index(self, logprobs, generated_text=None):
        """
        Find the index of the first token after "####".
        
        Args:
            logprobs: list of token logprobs
            generated_text: optional, the complete generated text
        
        Returns:
        int: index of the first token after "####", if not found, return None
        """
        # method 1: if there is generated text, search in the text directly
        if generated_text and "####" in generated_text:
            # find the position of "####" in the text
            hash_pos = generated_text.find("####")
            if hash_pos != -1:
                # calculate the number of characters after "####"
                text_after_hash = generated_text[hash_pos + 4:].lstrip()
                # need to find the corresponding token position
                # since the correspondence between token and character is complex, we use an approximate method
                # reconstruct the token sequence to locate
                return self._find_token_index_by_text(logprobs, text_after_hash[:20] if text_after_hash else "")
        
        # method 2: directly search "####" in the token sequence
        # "####" may be tokenized into a single token or multiple tokens
        # try to match different possible tokenization methods
        hash_patterns = [
            ["####"],  # single token
            ["#", "#", "#", "#"],  # four separate #
            ["##", "##"],  # two ##
            ["###", "#"],  # ### and #
        ]
        
        for i in range(len(logprobs)):
            for pattern in hash_patterns:
                if i + len(pattern) <= len(logprobs):
                    tokens = [logprobs[j].get('token', '') for j in range(i, i + len(pattern))]
                    # check if it matches (considering that token may contain spaces etc.)
                    tokens_clean = [t.replace('Ġ', ' ').replace('▁', ' ') for t in tokens]
                    if ''.join(tokens_clean).replace(' ', '') == '####':
                        # find the position after "####" (skip possible spaces)
                        next_idx = i + len(pattern)
                        # skip possible space tokens
                        while next_idx < len(logprobs) and logprobs[next_idx].get('token', '').strip() in ['', ' ', '\n']:
                            next_idx += 1
                        return next_idx
        
        return None
    
    def _find_token_index_by_text(self, logprobs, search_text):
        """
        Find the corresponding token index by the text fragment (approximate method).
        
        Args:
            logprobs: list of token logprobs
            search_text: text fragment to search
        
        Returns:
            int: index of the matched token, if not found, return None
        """
        if not search_text:
            return None
        
        # reconstruct the text of the token sequence
        reconstructed = ""
        for i, token_data in enumerate(logprobs):
            token = token_data.get('token', '')
            # handle special characters of the tokenizer
            token_clean = token.replace('Ġ', ' ').replace('▁', ' ')
            reconstructed += token_clean
            
            # check if it contains the search text
            if search_text in reconstructed:
                return i
        
        return None

    async def assess_CoT(self, data):
        if self.method == "entropy_based":
            assessment = self.entropy_based_assessment(data)
        elif self.method == "contrastive_guided":
            assessment = await self.contrastive_guided_assessment(data)
        elif self.method == "label_based":
            assessment = self.label_based_assessment(data)
        elif self.method == "self_guided":
            assessment = await self.self_guided_assessment(data)
        elif self.method == "Referi_score":
            assessment = await self.Referi_score_assessment(data)
        elif self.method == "CoT_WP":
            assessment = await self.CoT_WP_assessment(data)
        else:
            raise ValueError(f"Unsupported method: {self.method}")
        
        return assessment
