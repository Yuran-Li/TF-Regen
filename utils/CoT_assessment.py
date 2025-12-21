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
        # 仅当为列表且包含至少两条 CoT 才计算熵；否则视为已评估
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
        num_candidates = 10  # Stage 1 粗选数量
        num_good_examples = 1  # Stage 2 精选的 good 样本数量
        num_error_examples = 1  # Stage 2 精选的 error 样本数量
        
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
        """加载本地模型用于 Referi 评分计算"""
        model_path = self._resolve_model_path(self.model_name)
        self.referi_tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.referi_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        ).eval()
    
    def _resolve_model_path(self, model_name_or_path):
        """解析模型路径：将短名称转换为完整的 HuggingFace 路径"""
        # 检查是否是本地路径
        if os.path.exists(model_name_or_path) and os.path.isdir(model_name_or_path):
            return model_name_or_path
        
        # 模型名称映射表
        model_mapping = {
            "Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
            "Llama-3.1-70B-Instruct": "meta-llama/Llama-3.1-70B-Instruct",
            "Llama-3.3-70B-Instruct": "meta-llama/Llama-3.3-70B-Instruct",
        }
        
        # 如果找到映射，返回完整路径
        if model_name_or_path in model_mapping:
            return model_mapping[model_name_or_path]
        
        # 如果已经是完整路径（包含 /），直接返回
        if "/" in model_name_or_path:
            return model_name_or_path
        
        # 默认尝试添加 meta-llama/ 前缀
        return f"meta-llama/{model_name_or_path}"
    
    def _calc_loss(self, prompt, target_answer):
        """计算 target_answer 在 prompt 下的 CrossEntropy Loss"""
        input_text = prompt + target_answer
        inputs = self.referi_tokenizer(input_text, return_tensors="pt")
        
        # 将 inputs 移动到模型所在的设备
        if hasattr(self.referi_model, 'device'):
            device = self.referi_model.device
        elif hasattr(self.referi_model, 'hf_device_map'):
            device = next(iter(self.referi_model.hf_device_map.values()))['device']
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 构造 labels: mask 掉 prompt 部分，只计算 answer 的 loss
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
        num_candidates = 10  # Stage 1 粗选数量
        num_good_examples = 1  # Stage 2 精选的 good 样本数量
        num_error_examples = 1  # Stage 2 精选的 error 样本数量
        
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

        # 将 examples_list 转换为 few_shots 格式 (question, answer) 对
        few_shots = []
        for item in examples_list:
            question = item[config['question_field']]
            answer = item['eg'][0]
            few_shots.append((question, answer))
        
        if not few_shots:
            return "incorrect"  # 如果没有有效的 few_shots，返回 incorrect
        
        # 获取当前问题的 question 和 model_output
        question = data[config["question_field"]]
        model_output = data["CoT"]
        
        # 计算 Referi Score
        score = self._score_single_output(question, model_output, few_shots, config)
        
        # 根据阈值判断对错
        threshold = 0.0  # 如果 score > 0 (即降低了 Loss)，认为有帮助，判为 Correct
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
        
        # 系统提示词
        system_prompt = "Answer the given question step by step. And put the final answer after '####' in the end.\n"
        
        # 遍历每一个 few-shot 样本作为 Target
        for i, (target_q, target_a) in enumerate(few_shots):
            # 构造 Baseline Prompt (No Replace)
            context_str = system_prompt
            for j, (q, a) in enumerate(few_shots):
                if i == j: continue  # 跳过当前作为 Target 的样本
                context_str += f"Q: {q}\nA: {a}\n\n"
            
            baseline_prompt = f"{context_str}Q: {target_q}\nA:"
            
            # 构造 Referi Prompt (Replace)
            referi_prompt = f"{context_str}Q: {question}\nA: {model_output}\n\nQ: {target_q}\nA:"
            
            # 计算 Loss
            try:
                loss_baseline = self._calc_loss(baseline_prompt, target_a)
                loss_referi = self._calc_loss(referi_prompt, target_a)
                
                # Score > 0 意味着加上 model_output 后，target_a 的 loss 降低了（确信度提升）
                score = loss_baseline - loss_referi
                
                total_score += score
                valid_count += 1
                
            except Exception as e:
                print(f"Error calculating score for shot {i}: {e}")
                continue

        # 返回平均分
        if valid_count == 0:
            return -999.0
            
        return total_score / valid_count

    async def CoT_WP_assessment(self, data):
        """
        CoT-WP (Chain-of-Thought with Wrong Prediction) assessment.
        基于生成时的 Top-1 和 Top-2 token 概率差值来判断置信度。
        """
        # 检查是否有 logprobs 数据
        if 'logprobs' not in data or data['logprobs'] is None:
            # 如果没有 logprobs，触发重新生成以获取 logprobs
            logprobs = await self._regenerate_with_logprobs(data)
            if logprobs is None or not logprobs:
                return "incorrect"
        else:
            logprobs = data['logprobs']
            if not logprobs:
                return "incorrect"
        
        # 获取生成的文本（如果有的话，用于定位 "####"）
        generated_text = data.get("CoT", None)
        
        # 计算 CoT-WP 分数（只评估 "####" 之后的部分）
        score = self._calculate_cot_wp_score(logprobs, generated_text)
        
        # 根据阈值判断对错
        # 阈值需要根据实际情况调整，通常 score 越高表示置信度越高
        threshold = 0.1  # 可调整的阈值
        assessment = "correct" if score > threshold else "incorrect"
        return assessment
    
    async def _regenerate_with_logprobs(self, data):
        """
        重新生成 CoT 并获取 logprobs 数据。
        
        Returns:
            list: logprobs 列表，格式为 [{'token': '...', 'logprob': -0.1, 'top_logprobs': {...}}, ...]
                  如果生成失败或 API 不支持，返回 None
        """
        config = DATASET_CONFIG.get(self.dataset_name)
        if not config:
            return None
        
        # 构建 prompt（使用 zero-shot 模板）
        question = data[config["question_field"]]
        prompt = prompts.render_template(config.get("zero_shot_template"), question=question)
        
        # 调用 API 并获取 logprobs
        try:
            # 检查 API 类型
            if isinstance(self.api, models.OpenAIAPI):
                # OpenAI API 支持 logprobs
                response = await self.api.client.chat.completions.create(
                    model=self.api.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens,
                    logprobs=True,  # 启用 logprobs
                    top_logprobs=5,  # 返回 top 5 的 logprobs
                )
                
                # 提取 logprobs
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
                # vLLM API 也支持 logprobs
                # 使用 backoff 重试机制处理连接错误
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
                    # 静默处理错误，不打印警告（因为已经在重试中处理了）
                    return None
            else:
                # 对于其他 API 类型，尝试通用方式
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
                        # 静默处理，不打印警告
                        return None
        except Exception as e:
            # 只在最终失败时打印错误
            return None
        
        return None
    
    def _calculate_cot_wp_score(self, logprobs, generated_text=None):
        """
        计算 CoT-WP 分数：基于 Top-1 和 Top-2 token 概率的平均差值。
        只评估 "####" 之后的部分（答案部分）。
        
        Args:
            logprobs: list of dicts, each dict contains 'top_logprobs' for a token.
                     Format: [{'token': '2', 'logprob': -0.1, 'top_logprobs': {'2': -0.1, '3': -2.5, ...}}, ...]
            generated_text: 可选，生成的完整文本，用于定位 "####" 的位置
        
        Returns:
            float: 平均概率差值，值越大表示置信度越高
        """
        if not logprobs:
            return 0.0
        
        # 找到 "####" 之后的部分
        answer_start_idx = self._find_answer_start_index(logprobs, generated_text)
        
        # 只使用 "####" 之后的 logprobs
        if answer_start_idx is not None and answer_start_idx < len(logprobs):
            target_tokens = logprobs[answer_start_idx:]
        else:
            # 如果没有找到 "####"，直接返回 0.0（会被判定为 incorrect）
            return 0.0
        
        if not target_tokens:
            return 0.0
        
        diffs = []
        
        for token_data in target_tokens:
            # 获取 Top-2 的 logprobs
            # OpenAI API 返回的是 logprob，需要 exp 转回概率
            if 'top_logprobs' not in token_data or not token_data['top_logprobs']:
                continue
                
            top_probs = sorted([np.exp(lp) for lp in token_data['top_logprobs'].values()], reverse=True)
            
            if len(top_probs) >= 2:
                p_top1 = top_probs[0]
                p_top2 = top_probs[1]
                diffs.append(p_top1 - p_top2)
        
        if not diffs:
            return 0.0
        
        # 计算平均差值作为最终分数
        return np.mean(diffs)
    
    def _find_answer_start_index(self, logprobs, generated_text=None):
        """
        找到 "####" 之后第一个 token 的索引。
        
        Args:
            logprobs: token logprobs 列表
            generated_text: 可选，生成的完整文本
        
        Returns:
            int: "####" 之后第一个 token 的索引，如果没找到返回 None
        """
        # 方法1: 如果有生成的文本，直接在文本中查找
        if generated_text and "####" in generated_text:
            # 找到 "####" 在文本中的位置
            hash_pos = generated_text.find("####")
            if hash_pos != -1:
                # 计算 "####" 之后的字符数
                text_after_hash = generated_text[hash_pos + 4:].lstrip()
                # 需要找到对应的 token 位置
                # 由于 token 和字符的对应关系复杂，我们使用近似方法
                # 通过重建 token 序列来定位
                return self._find_token_index_by_text(logprobs, text_after_hash[:20] if text_after_hash else "")
        
        # 方法2: 直接在 token 序列中查找 "####"
        # "####" 可能被 tokenize 为单个 token 或多个 token
        # 尝试匹配不同的可能 tokenization 方式
        hash_patterns = [
            ["####"],  # 单个 token
            ["#", "#", "#", "#"],  # 四个单独的 #
            ["##", "##"],  # 两个 ##
            ["###", "#"],  # ### 和 #
        ]
        
        for i in range(len(logprobs)):
            for pattern in hash_patterns:
                if i + len(pattern) <= len(logprobs):
                    tokens = [logprobs[j].get('token', '') for j in range(i, i + len(pattern))]
                    # 检查是否匹配（考虑 token 可能包含空格等）
                    tokens_clean = [t.replace('Ġ', ' ').replace('▁', ' ') for t in tokens]
                    if ''.join(tokens_clean).replace(' ', '') == '####':
                        # 找到 "####" 之后的位置（跳过可能的空格）
                        next_idx = i + len(pattern)
                        # 跳过可能的空格 token
                        while next_idx < len(logprobs) and logprobs[next_idx].get('token', '').strip() in ['', ' ', '\n']:
                            next_idx += 1
                        return next_idx
        
        return None
    
    def _find_token_index_by_text(self, logprobs, search_text):
        """
        通过文本片段找到对应的 token 索引（近似方法）。
        
        Args:
            logprobs: token logprobs 列表
            search_text: 要搜索的文本片段
        
        Returns:
            int: 匹配的 token 索引，如果没找到返回 None
        """
        if not search_text:
            return None
        
        # 重建 token 序列的文本
        reconstructed = ""
        for i, token_data in enumerate(logprobs):
            token = token_data.get('token', '')
            # 处理 tokenizer 的特殊字符
            token_clean = token.replace('Ġ', ' ').replace('▁', ' ')
            reconstructed += token_clean
            
            # 检查是否包含搜索文本
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