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
from .openai_embedder import OpenAIEmbedder
# Fix OpenBLAS warning by setting thread limits before using SentenceTransformer
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
from .check import check_answer_gsm_hard, check_answer_strategyqa, check_answer_gsm8k, check_answer_math,check_answer_coin, check_answer_letter, check_answer_legal
from .quality import get_prediction
# unified dataset configuration
DATASET_CONFIG = {
    "gsm8k": {
        "question_field": "question",
        "template": "few_shot_CoT_generation_gsm8k_prompt",
        "template_rectify": "few_shot_CoT_generation_gsm8k_prompt(rectify)",
        "template_rectify_stage1": "few_shot_CoT_generation_gsm8k_prompt(rectify-stage1)",
        "template_rectify_stage2": "few_shot_CoT_generation_gsm8k_prompt(rectify-stage2)",
        "template_AoT": "few_shot_CoT_generation_gsm8k_prompt(AoT)",
        "zero_shot_template": "./zero_shot/CoT_generation_gsm8k_prompt",
        "fixed_template": "few_shot_CoT_generation_gsm8k_prompt(contrast-cot)",
        "check_func": check_answer_gsm8k,
        "answer_field": "answer"
    },
    "gsm_hard": {
        "question_field": "input", 
        "template": "few_shot_CoT_generation_gsm8k_prompt",
        "zero_shot_template": "./zero_shot/CoT_generation_gsm8k_prompt",
        "check_func": check_answer_gsm_hard,
        "answer_field": "target"
    },
    "strategyQA": {
        "question_field": "question",
        "template": "few_shot_CoT_generation_strategyQA_prompt", 
        "zero_shot_template": "./zero_shot/CoT_generation_strategyQA_prompt",
        "fixed_template": "few_shot_CoT_generation_strategyQA_prompt(01-llama70B)",
        "check_func": check_answer_strategyqa,
        "answer_field": "answer"
    },
    "math": {
        "question_field": "problem",
        "template": "few_shot_CoT_generation_gsm8k_prompt",
        "zero_shot_template": "./zero_shot/CoT_generation_gsm8k_prompt",
        "check_func": check_answer_math,
        "answer_field": "answer"
    },
    "coin": {
        "question_field": "inputs",
        "template": "few_shot_CoT_generation_coin_prompt",
        "zero_shot_template": "./zero_shot/CoT_generation_coin_prompt",
        "check_func": check_answer_coin,
        "answer_field": "targets"
    },
    "letter": {
        "question_field": "question", 
        "template": "few_shot_CoT_generation_letter_prompt",
        "zero_shot_template": "./zero_shot/CoT_generation_letter_prompt",
        "check_func": check_answer_letter,
        "answer_field": "answer"
    },
    "legal": {
        "question_field": "text",
        "template": "few_shot_CoT_generation_legal_abercrombie_prompt",
        "zero_shot_template": "./zero_shot/CoT_generation_legal_abercrombie_prompt",
        "check_func": check_answer_legal,
        "answer_field": "answer"
    }
}

def split_assessment(assessment: str) -> tuple[str, str]:
        # split the assessment into analysis and corrected reasoning
        parts = assessment.split("## Analysis")
        if len(parts) < 2:
            return "", ""  # if the analysis part is not found, return empty strings
        
        # get the content after the analysis part
        remaining = parts[1]
        
        # split the corrected reasoning part
        parts = remaining.split("## Corrected Reasoning")
        if len(parts) < 2:
            return remaining.strip(), ""  # if the corrected reasoning part is not found, return the entire analysis part
        
        analysis = parts[0].strip()
        corrected = parts[1].strip()
        
        return analysis, corrected

class CoTGeneration:
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
        
        # Pre-load sentence transformer model and embeddings for similarity method
        if self.method == "similarity" and self.eg_data is not None:
            self.sentence_model = SentenceTransformer("all-mpnet-base-v2", device="cpu") #all-MiniLM-L6-v2,all-mpnet-base-v2
            self.eg_embeddings = self.sentence_model.encode([item["question"] for item in self.eg_data])
        elif self.method == "contrastive" or self.method == "contrastive_rectify" and self.eg_data is not None:
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
                        # check_funcs = {self.dataset_name: lambda: (config["check_func"](corrected_1, item[config["answer_field"]]) and config["check_func"](corrected_2, item[config["answer_field"]]))}
                        check_funcs = {self.dataset_name: lambda: (config["check_func"](corrected_1, item[config["answer_field"]]))}
                    
                    else:
                        check_funcs = {}
                    if self.dataset_name in check_funcs and check_funcs[self.dataset_name]():
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
            self.sentence_model = SentenceTransformer("all-mpnet-base-v2", device="cpu")
            ## use OpenAI embedding model
            # self.sentence_model = OpenAIEmbedder("text-embedding-3-large")
            # self.sentence_model = SentenceTransformer("facebook/contriever", device="cpu")
            self.eg_embeddings = self.sentence_model.encode([item[field_mapping[self.dataset_name]] for item in self.filtered_items])
            self.good_eg_embeddings = self.sentence_model.encode([item[field_mapping[self.dataset_name]] for item in self.good_items])
            self.error_eg_embeddings = self.sentence_model.encode([item[field_mapping[self.dataset_name]] for item in self.error_items])
            '''
            embedding method 2:
            example based similarity
            select examples with helpful questions, reasoning and answer
            '''
            # self.sentence_model = SentenceTransformer("facebook/contriever", device="cpu") #facebook/contriever-msmarco
            # self.sentence_model = OpenAIEmbedder("text-embedding-3-large")
            # self.sentence_model = SentenceTransformer("all-mpnet-base-v2", device="cpu")
            # self.good_eg_embeddings = self.sentence_model.encode(
            #     [
            #         f"Question:{item[field_mapping[self.dataset_name]]}\n"
            #         f"Answer:{item.get('eg')}"
            #         for item in self.good_items
            #     ],
            #     normalize_embeddings=True,
            # )
            # self.error_eg_embeddings = self.sentence_model.encode(
            #     [
            #         f"Question:{item[field_mapping[self.dataset_name]]}\n"
            #         f"Answer:{split_assessment(item['assessment'].get('rectify'))[1]}\n"
            #         for item in self.error_items
            #     ],
            #     normalize_embeddings=True,
            # )

    def generate_cot_zero_shot(self, data):
        # use unified config for zero-shot generation
        config = DATASET_CONFIG.get(self.dataset_name)
        if not config:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
        
        question = data[config["question_field"]]
        prompt = prompts.render_template(config["zero_shot_template"], question=question)
        return prompt

    def generate_cot_fixed(self, data):
        # use unified config for fixed generation
        config = DATASET_CONFIG.get(self.dataset_name)
        if not config or "fixed_template" not in config:
            raise ValueError(f"Unsupported dataset or no fixed template: {self.dataset_name}")
        
        question = data[config["question_field"]]
        prompt = prompts.render_template(config["fixed_template"], question=question)
        return prompt

    def generate_cot_random(self, data):
        if self.eg_data is None:
            raise ValueError("Example data is required for 'random' method. Please provide eg_path parameter.")
        random.seed(self.seed)
        # ## random select 4 items
        selected_items = random.sample(self.eg_data, min(4, len(self.eg_data)))
        
        if self.dataset_name == "gsm8k":
            # extract question and eg pairs to form complete examples
            examples_list = []
            for item in selected_items:
                example = f"Question: {item['question']}\nAnswer:{item['eg']}"
                # print(example)
                examples_list.append(example)
            examples = "\n\n".join(examples_list)
            prompt = prompts.render_template(
                "few_shot_CoT_generation_gsm8k_prompt", question=data["question"], examples=examples)
        elif self.dataset_name == "gsm_hard":
            # extract question and eg pairs to form complete examples
            examples_list = []
            for item in selected_items:
                example = f"Question: {item['input']}\nAnswer:{item['eg']}"
                # print(example)
                examples_list.append(example)
            examples = "\n\n".join(examples_list)
            prompt = prompts.render_template(
                "few_shot_CoT_generation_gsm8k_prompt", question=data["input"], examples=examples)
        elif self.dataset_name == "strategyQA":
            # extract question and eg pairs to form complete examples
            examples_list = []
            for item in selected_items:
                example = f"Question: {item['question']} true/false?\nAnswer:{item['eg']}"
                examples_list.append(example)
            examples = "\n\n".join(examples_list)
            prompt = prompts.render_template(
                "few_shot_CoT_generation_strategyQA_prompt", question=data["question"], examples=examples)
        return prompt

    def generate_cot_similarity(self, data):
        if self.eg_data is None:
            raise ValueError("Example data is required for 'similarity' method. Please provide eg_path parameter.")
        # Use pre-loaded model and embeddings for better performance
        question_embedding = self.sentence_model.encode(data["question"]).reshape(1, -1)
        similarities = cosine_similarity(question_embedding, self.eg_embeddings)
        most_similar_index = np.argsort(similarities[0])[-4:][::-1] # top 4
        most_similar_egs = [self.eg_data[i] for i in most_similar_index]
        examples_list = []
        for eg in most_similar_egs:
            example = f"Question: {eg['question']}\nAnswer:{eg['eg']}"
            examples_list.append(example)
        examples = "\n\n".join(examples_list)
        if self.dataset_name == "gsm8k":
            prompt = prompts.render_template(
                "few_shot_CoT_generation_gsm8k_prompt", question=data["question"], examples=examples)
        elif self.dataset_name == "mmlu":
            prompt = prompts.render_template(
                "few_shot_CoT_generation_mmlu_prompt", question=data["question"], examples=examples)
        elif self.dataset_name == "strategyQA":
            prompt = prompts.render_template(
                "few_shot_CoT_generation_strategyQA_prompt", question=data["question"], examples=examples)
        return prompt

    def generate_cot_contrastive(self, data):
        selected_items = []
        if self.eg_data is None:
            raise ValueError("Example data is required for 'contrastive' method. Please provide eg_path parameter.")
        # set seed for random sampling to ensure reproducibility
        random.seed(self.seed)

        ## random select items
        # selected_items.extend(random.sample(self.filtered_items, min(4, len(self.filtered_items))))

        ## select items from good and error items separately
        # selected_items.extend(random.sample(self.good_items, min(2, len(self.good_items))))
        # selected_items.extend(random.sample(self.error_items, min(1, len(self.error_items))))

        ## similarity select items
        ## use unified config for field mapping
        field_mapping = {dataset: config["question_field"] for dataset, config in DATASET_CONFIG.items()}
        # query_text = f"Question: {data[field_mapping[self.dataset_name]]}"
        # question_embedding = self.sentence_model.encode(query_text, normalize_embeddings=True).reshape(1, -1)
        question_embedding = self.sentence_model.encode(data[field_mapping[self.dataset_name]], normalize_embeddings=True).reshape(1, -1)
        
        ## select similar items from filtered items
        # similarities = cosine_similarity(question_embedding, self.eg_embeddings)
        # most_similar_index = np.argsort(similarities[0])[-4:][::-1] # top 4
        # selected_items.extend([self.filtered_items[i] for i in most_similar_index])
        ## select similar items from good and error items separately
        similarities = cosine_similarity(question_embedding, self.good_eg_embeddings)
        most_similar_index = np.argsort(similarities[0])[-1:][::-1] # top 1
        selected_items.extend([self.good_items[i] for i in most_similar_index])
        similarities = cosine_similarity(question_embedding, self.error_eg_embeddings)
        most_similar_index = np.argsort(similarities[0])[-1:][::-1] # top 1
        selected_items.extend([self.error_items[i] for i in most_similar_index])
        
        config = DATASET_CONFIG.get(self.dataset_name)
        if not config:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
        
        # generate examples
        correct_examples_list = []
        prefix = "Note: The following example contains an incorrect answer followed by its analysis and correction. Do not imitate the incorrect answer."
        incorrect_examples_list = [prefix]
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
                principles = item["assessment"].get("principles")
                error_tag = item["assessment"].get("error_tag")
                has_reflection = True
            
            if item["assessment"] == "bad":
                continue
            elif item["assessment"] == "good":
                example = f"Question: {item[config['question_field']]}\nCorrect Answer:{prediction}\n\n"
                correct_examples_list.append(example)
            elif has_reflection:
                example = f"Question: {item[config['question_field']]}\nIncorrect Answer:{prediction}\nAnalysis:{principles}\nCorrect Answer:{corrected_1}\n\n"
                incorrect_examples_list.append(example)
            else:  # improved response
                # example = f"Question: {item[config['question_field']]}\nCorrect Answer:{item['assessment']}\n\n"
                raise ValueError(f"Invalid assessment: {item['assessment']}")
        examples = "\n\n".join(correct_examples_list + incorrect_examples_list)
        prompt = prompts.render_template(
            config["template"], 
            question=data[config["question_field"]], 
            examples=examples
        )
        # print(f"prompt: {prompt}")
        return prompt
    
    async def generate_cot_contrastive_rectify(self, data):
        selected_items = []
        if self.eg_data is None:
            raise ValueError("Example data is required for 'contrastive' method. Please provide eg_path parameter.")
        # set seed for random sampling to ensure reproducibility
        random.seed(self.seed)

        ## random select items
        # selected_items.extend(random.sample(self.filtered_items, min(4, len(self.filtered_items))))

        ## select items from good and error items separately
        # selected_items.extend(random.sample(self.good_items, min(2, len(self.good_items))))
        # selected_items.extend(random.sample(self.error_items, min(1, len(self.error_items))))

        ## similarity select items
        ## use unified config for field mapping
        field_mapping = {dataset: config["question_field"] for dataset, config in DATASET_CONFIG.items()}
        # query_text = f"Question: {data[field_mapping[self.dataset_name]]}"
        # question_embedding = self.sentence_model.encode(query_text, normalize_embeddings=True).reshape(1, -1)
        question_embedding = self.sentence_model.encode(data[field_mapping[self.dataset_name]], normalize_embeddings=True).reshape(1, -1)
        
        ## select similar items from filtered items
        # similarities = cosine_similarity(question_embedding, self.eg_embeddings)
        # most_similar_index = np.argsort(similarities[0])[-4:][::-1] # top 4
        # selected_items.extend([self.filtered_items[i] for i in most_similar_index])
        ## select similar items from good and error items separately
        # similarities = cosine_similarity(question_embedding, self.good_eg_embeddings)
        # most_similar_index = np.argsort(similarities[0])[-2:][::-1] # top 2
        # selected_items.extend([self.good_items[i] for i in most_similar_index])
        similarities = cosine_similarity(question_embedding, self.error_eg_embeddings)
        most_similar_index = np.argsort(similarities[0])[-3:][::-1] # top 3
        selected_items.extend([self.error_items[i] for i in most_similar_index])
        
        config = DATASET_CONFIG.get(self.dataset_name)
        if not config:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
        
        # return self.one_stage_prompt(data, selected_items, config)
        return await self.two_stage_prompt(data, selected_items, config)

    def one_stage_prompt(self, data, selected_items, config):
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
                principles = item["assessment"].get("principles")
                has_reflection = True
            
            if item["assessment"] == "bad":
                continue
            elif item["assessment"] == "good":
                example = f"Question: {item[config['question_field']]}\nCorrect Answer:{prediction}\n\n"
            elif has_reflection:
                example = f"Question: {item[config['question_field']]}\nIncorrect Answer:{prediction}\n## Analysis:{principles}\n## Correct Answer:{corrected_1}\n\n"
            else:  # improved response
                # example = f"Question: {item[config['question_field']]}\nCorrect Answer:{item['assessment']}\n\n"
                raise ValueError(f"Invalid assessment: {item['assessment']}")
            examples_list.append(example)
        
        examples = "\n\n".join(examples_list)
        # print(f"examples: {examples}")
        prompt = prompts.render_template(
            config["template_rectify"], 
            question=data[config["question_field"]], 
            prediction=data["CoT"],
            examples=examples
        )
        # print(f"prompt: {prompt}")
        return prompt
    
    async def two_stage_prompt(self, data, selected_items, config):
        # generate examples
        reflection_examples_list = []
        rectify_examples_list = []
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
                principles = item["assessment"].get("principles")
                has_reflection = True
            
            if item["assessment"] == "bad":
                continue
            elif item["assessment"] == "good":
                example = f"Question: {item[config['question_field']]}\nCorrect Answer:{prediction}\n\n"
            elif has_reflection:
                reflection_example = f"Question: {item[config['question_field']]}\nIncorrect Answer:{prediction}\nReflection:{analysis_2}\n\n"
                rectify_example = f"Question: {item[config['question_field']]}\nIncorrect Answer:{prediction}\nReflection:{analysis_2}\nCorrect Answer:{corrected_2}\n\n"
            else:  # improved response
                # example = f"Question: {item[config['question_field']]}\nCorrect Answer:{item['assessment']}\n\n"
                raise ValueError(f"Invalid assessment: {item['assessment']}")
            reflection_examples_list.append(reflection_example)
            rectify_examples_list.append(rectify_example)
        
        reflection_examples = "\n\n".join(reflection_examples_list)
        rectify_examples = "\n\n".join(rectify_examples_list)
        reflection_prompt = prompts.render_template(
            config["template_rectify_stage1"], 
            question=data[config["question_field"]], 
            prediction=data["CoT"],
            examples=reflection_examples
        )
        # print(f"reflection_prompt: {reflection_prompt}")
        reflection_output = await self.api.chat(
                        messages=[{"role": "user", "content": reflection_prompt}],
                        temperature=self.temperature,
                        top_p=self.top_p,
                        max_tokens=self.max_tokens,
                        seed=self.seed,
                    )
        rectify_prompt = prompts.render_template(
            config["template_rectify_stage2"], 
            question=data[config["question_field"]], 
            prediction=data["CoT"],
            reflection=reflection_output,
            examples=rectify_examples
        )
        # print(f"rectify_prompt: {rectify_prompt}")
        return rectify_prompt
   
    async def generate_CoT(self, data):
        if self.method == "zero_shot":
            prompt = self.generate_cot_zero_shot(data)
        elif self.method == "fixed":
            prompt = self.generate_cot_fixed(data)
        elif self.method == "random":
            prompt = self.generate_cot_random(data)
        elif self.method == "similarity":
            prompt = self.generate_cot_similarity(data)
        elif self.method == "contrastive":
            prompt = self.generate_cot_contrastive(data)
        elif self.method == "contrastive_rectify":
            prompt = await self.generate_cot_contrastive_rectify(data)
        else:
            raise ValueError(f"Unsupported method: {self.method}")
        if self.temperature == 0.0:
            output = await self.api.chat(
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.temperature,
                        top_p=self.top_p,
                        max_tokens=self.max_tokens,
                        seed=self.seed,
                    )
        else:
            output = []
            for _ in range(3):
                output.append(await self.api.chat(
                            messages=[{"role": "user", "content": prompt}],
                            temperature=self.temperature,
                            top_p=self.top_p,
                            max_tokens=self.max_tokens,
                            seed=self.seed,
                        ))
        
        return output


from collections import Counter
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")

TOKEN_LIMIT = 400

async def improve_contrastive_example(dataset_name, eg_path, model_name, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    enc = tiktoken.get_encoding("cl100k_base")

    def trigram_repetition_ratio(text: str, min_len=3):
        toks = text.split()
        if len(toks) < 3: return 0.0
        trigrams = [" ".join(toks[i:i+3]) for i in range(len(toks)-2)]
        cnt = Counter(trigrams)
        rep = sum(c for c in cnt.values() if c > 1)
        return rep / max(1, len(trigrams))

    def is_noisy(eg: str) -> bool:
        lines = eg.splitlines()
        long_line = any(len(l) > 300 for l in lines)
        many_lines = len(lines) > 20
        repeated = trigram_repetition_ratio(eg) > 0.2
        
        heavy_ops = sum(ch in "=/*+-" for ch in eg) / max(1, len(eg)) > 0.08
        
        noisy_phrases = ["we already did that", "assume", "fraction of the total time", "inequality", "However,"]
        noisy_hits = sum(eg.count(p) for p in noisy_phrases) >= 3
        return long_line or repeated or heavy_ops or noisy_hits
    
    try:
        with open(eg_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return
    
    try:
        api = models.get_chat_api_from_model(model_name)
    except Exception as e:
        print(f"Error getting chat API: {e}")
        return

    for item in tqdm(data):
        # use unified config for field extraction
        config = DATASET_CONFIG.get(dataset_name)
        if not config:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        question = item[config["question_field"]]
        answer = item[config["answer_field"]]
        eg = item["eg"][-1]
        assessment = item["assessment"]
        
        has_reflection = False
        if isinstance(item["assessment"], dict):
                analysis_1, corrected_1 = split_assessment(item["assessment"].get("rectify"))
                analysis_2 = item["assessment"].get("reflect")
                corrected_2 = item["assessment"].get("reflective_rectify")
                principles = item["assessment"].get("principles")
                has_reflection = True

        if has_reflection:
            """only denoise"""
            # need_improve = (len(enc.encode(eg)) > TOKEN_LIMIT) or is_noisy(eg)
            # if need_improve:
            #     # print(f"need to improve incorrect answer: {eg}")
            #     improve_contrastive_example_prompt = prompts.render_template(
            #         "improve_contrastive_example_prompt",
            #         question=question,
            #         answer=answer,
            #         eg=eg,
            #         analysis=analysis_1,
            #         corrected=corrected_1
            #     )
            #     improved_eg = await api.chat(
            #         messages=[{"role": "user", "content": improve_contrastive_example_prompt}],
            #         temperature=0.0,
            #         top_p=1.0,
            #         max_tokens=1024,
            #     )
            #     # print(f"improved eg: {improved_eg}")
            #     item["eg"] = improved_eg
            """rewrite the incorrect answer more contrastive ICL friendly"""
            # print(f"need to rewrite incorrect answer: {eg}")
            rewrite_contrastive_example_prompt = prompts.render_template(
                    "rewrite_contrastive_example_prompt",
                    question=question,
                    answer=answer,
                    eg=eg,
                    analysis=analysis_1,
                    corrected=corrected_1
            )
            improved_eg = await api.chat(
                messages=[{"role": "user", "content": rewrite_contrastive_example_prompt}],
                temperature=0.0,
                top_p=1.0,
                max_tokens=1024,
            )
            # print(f"improved eg: {improved_eg}")
            item["eg"] = improved_eg
    
    with open(f"output/{model_name}_{dataset_name}_improved_eg.json", "w", encoding="utf-8") as f:
        print(f"Saving improved eg to output/{model_name}_{dataset_name}_improved_eg.json")
        json.dump(data, f, ensure_ascii=False, indent=2)
