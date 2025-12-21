import re
import asyncio
from typing import Dict, Any, Optional, Union
import utils.prompts as prompts
import utils.models as models
from .check import check_answer_strategyqa, check_answer_gsm8k, check_answer_gsm_hard, check_answer_math, check_answer_coin, check_answer_legal, check_answer_letter, check_answer_bamboogle, check_answer_Headline

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

# define constants
QUALITY_THRESHOLD = 7
MAX_IMPROVEMENT_ATTEMPTS = 3

# dataset field mapping configuration
DATASET_CONFIG = {
    "gsm_hard": {
        "question": "input",
        "label": "target", 
        "gold_reasoning": "code",
        "prediction": "eg"
    },
    "gsm8k": {
        "question": "question",
        "label": "answer",
        "prediction": "eg"
    },
    "math": {
        "question": "problem",
        "label": "answer",
        "gold_reasoning": "solution",
        "prediction": "eg"
    },
    "strategyQA": {
        "question": "question",
        "label": "answer",
        "gold_reasoning": lambda d: f"Decomposition:\n{d['decomposition'][0]}\nFacts:\n{d['facts'][0]}",
        "prediction": "eg"
    },
    "bamboogle": {
        "question": "Question",
        "label": "Answer",
        "prediction": "eg"
    },
    "coin": {
        "question": "inputs",
        "label": "targets",
        "prediction": "eg"
    },
    "letter": {
        "question": "question",
        "label": "answer",
        "prediction": "eg"
    },
    "legal": {
        "question": "text",
        "label": "answer",
        "prediction": "eg"
    },
    "Headline": {
        "question": "News",
        "label": "Price Sentiment",
        "prediction": "eg"
    }
}

def get_prediction(predictions: list[str], label: str, dataset_name: str) -> tuple[str, int]:
        # Check answer correctness
        check_funcs = {
            "strategyQA": check_answer_strategyqa,
            "gsm8k": check_answer_gsm8k,
            "gsm_hard": check_answer_gsm_hard,
            "math": check_answer_math,
            "coin": check_answer_coin,
            "letter": check_answer_letter,
            "legal": check_answer_legal,
            "bamboogle": check_answer_bamboogle,
            "Headline": check_answer_Headline
        }
        check_func = check_funcs.get(dataset_name)
        if not check_func:
            return f"error: unsupported dataset {dataset_name}"
        
        correct_flags = [check_func(prediction, label) for prediction in predictions]
        is_correct = sum(1 for f in correct_flags if f)
        if predictions:
            if is_correct == 0 or is_correct == len(predictions):
                prediction = predictions[-1]
            else:
                # mixed: pick the last correct prediction
                for i in range(len(predictions) - 1, -1, -1):
                    if correct_flags[i]:
                        prediction = predictions[i]
                        break
        return prediction, is_correct

class QualityAssessment:
    """Assessment of answer quality and improvement suggestions"""
    
    def __init__(self, model_name: str, model_type: str, dataset_name: str):
        self.model_name = model_name
        self.model_type = model_type
        self.dataset_name = dataset_name
        if model_type == "api":
            self.api = models.get_chat_api_from_model(self.model_name)
        else:
            self.api = None

    def _extract_data_fields(self, data: Dict[str, Any]) -> tuple:
        """
        extract fields from data
        Args:
            data: original data dictionary
        Returns:
            tuple: (question, label, prediction, gold_reasoning)
        """
        config = DATASET_CONFIG.get(self.dataset_name)
        if not config:
            raise ValueError(f"Unsupported dataset {self.dataset_name}")
        
        # extract basic fields
        question = data[config["question"]]
        label = data[config["label"]]
        prediction = data[config["prediction"]]
        
        # process gold_reasoning (if exists)
        gold_reasoning = None
        if "gold_reasoning" in config:
            if callable(config["gold_reasoning"]):
                gold_reasoning = config["gold_reasoning"](data)
            else:
                gold_reasoning = data[config["gold_reasoning"]]
        
        return question, label, prediction, gold_reasoning

    async def score_response(self, question: str, prediction: str, label: str) -> float:
        """
        Score the response quality
        Args:
            question: The original question
            prediction: The predicted answer
            label: The correct answer
        Returns:
            float: Quality score (0-10)
        """
        try:
            score_prompt = prompts.render_template(
                # "score_prompt", 
                "score_prompt_with_gold_label",
                question=question, 
                label=label, 
                prediction=prediction
            )
            # print(f"score_prompt: {score_prompt}")
            score_response = await self.api.chat(
                messages=[{"role": "user", "content": score_prompt}],
                temperature=0.0,
                top_p=1.0,
                max_tokens=1024,
            )
            return float(score_response)
        except (ValueError, TypeError) as e:
            print(f"Error scoring response: {e}")
            return 0.0

    async def improve_response(self, question: str, prediction: str, label: str) -> str:
        """
        Try to improve the response
        Args:
            question: The original question
            prediction: The predicted answer
            label: The correct answer
        Returns:
            str: Improved response or "bad" if improvement failed
        """
        for _ in range(MAX_IMPROVEMENT_ATTEMPTS):
            try:
                improve_prompt = prompts.render_template(
                    # "improve_prompt",
                    "improve_prompt_with_gold_label",
                    question=question,
                    label=label,
                    prediction=prediction
                )
                improve_response = await self.api.chat(
                    messages=[{"role": "user", "content": improve_prompt}],
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=1024,
                )

                compare_prompt = prompts.render_template(
                    "compare_prompt",
                    question=question,
                    label=label,
                    prediction=prediction,
                    improve_response=improve_response
                )
                is_better = await self.api.chat(
                    messages=[{"role": "user", "content": compare_prompt}],
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=1024,
                )
                
                if is_better.lower() in ['yes', 'true', '1']:
                    # print("improved response is better")
                    # print(f"improved response: {improve_response}")
                    return improve_response
            except Exception as e:
                print(f"Error improving response: {e}")
                continue
        return "bad"

    async def rectify_response(self, question: str, prediction: str, label: str) -> str:
        """
        Rectify an incorrect response
        Args:
            question: The original question
            prediction: The predicted answer
            label: The correct answer
        Returns:
            str: Rectified response
        """
        try:
            rectify_prompt = prompts.render_template(
                "rectify_prompt",
                question=question,
                label=label,
                prediction=prediction
            )
            # print(f"rectify_prompt: {rectify_prompt}")
            rectified_response = await self.api.chat(
                messages=[{"role": "user", "content": rectify_prompt}],
                temperature=0.0,
                top_p=1.0,
                max_tokens=1024,
            )
            # print(f"rectified response: {rectified_response}")
            return rectified_response
        except Exception as e:
            print(f"Error rectifying response: {e}")
            return "error in rectification"

    async def reflection_and_rectify_response(self, question: str, prediction: str, label: str) -> Dict[str, Any]:
            reflect_rectify_response = {}
            try:
                ## 1. directly rectify
                rectify_prompt = prompts.render_template(
                    "./reflection/rectify_prompt(Headline)",
                    question=question,
                    prediction=prediction,
                    label=label
                )
                rectify_response = await self.api.chat(
                    messages=[{"role": "user", "content": rectify_prompt}],
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=1024,
                )
                reflect_rectify_response["rectify"] = rectify_response
                analysis, rectify = split_assessment(rectify_response)
                ## 3. reflection
                reflect_prompt = prompts.render_template(
                    "./reflection/reflection_prompt",
                    question=question,
                    prediction=prediction,
                    label=label
                )
                reflect_response = await self.api.chat(
                    messages=[{"role": "user", "content": reflect_prompt}],
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=1024,
                )
                reflect_rectify_response["reflect"] = reflect_response
                # ## rectify based on reflection
                # rectify_prompt = prompts.render_template(
                #     "./reflection/reflective_rectify_prompt",
                #     question=question,
                #     prediction=prediction,
                #     reflection=reflect_response
                # )
                # rectify_response = await self.api.chat(
                #     messages=[{"role": "user", "content": rectify_prompt}],
                #     temperature=0.0,
                #     top_p=1.0,
                #     max_tokens=1024,
                # )
                # reflect_rectify_response["reflective_rectify"] = rectify_response
                ## 4. learn principles from mistakes
                principle_prompt = prompts.render_template(
                    "./reflection/principle_prompt",
                    question=question,
                    prediction=prediction,
                    reflection=reflect_response
                )
                principle_response = await self.api.chat(
                    messages=[{"role": "user", "content": principle_prompt}],
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=1024,
                )
                reflect_rectify_response["principles"] = principle_response
                ## 5. error tag
                error_tag_prompt = prompts.render_template(
                    "./reflection/error_tag_prompt",
                    reflection=reflect_response
                )
                error_tag_response = await self.api.chat(
                    messages=[{"role": "user", "content": error_tag_prompt}],
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=1024,
                )
                reflect_rectify_response["error_tag"] = error_tag_response
                return reflect_rectify_response
            except Exception as e:
                print(f"Error rectifying response: {e}")
                return "error in rectification"

    async def quality_assessment(self, data: Dict[str, Any]) -> Union[str, Dict[str, Any]]:
        """
        Assess the quality of a response and improve if needed
        Args:
            data: Dictionary containing question, answer and prediction
        Returns:
            str: Assessment result ("good", improved response, or rectified response)
        """
        try:
            question, label, predictions, gold_reasoning = self._extract_data_fields(data)
        except (KeyError, ValueError) as e:
            print(f"Error extracting data fields: {e}")
            return "error: incomplete data"

        prediction, is_correct = get_prediction(predictions, label, self.dataset_name)
        if self.dataset_name == "gsm_hard" or self.dataset_name == "math" or self.dataset_name == "strategyQA":
            label = gold_reasoning + "####" + str(label)
        if is_correct > 0: ## add positive examples for uncertain cases, where include correct and incorrect predictions
            score = await self.score_response(question, prediction, label)
            if score > QUALITY_THRESHOLD:
                return "good"
            else:
                return "bad"
                #return await self.improve_response(question, prediction, label) ##no response improvement for the sake of avoiding hallucination
        elif is_correct == 0:
            ## method 1: analyze and rectify in a single call
            # return await self.rectify_response(question, prediction, label)
            ## method 2: first call analyze, give rectification plans; then second call rectify
            return await self.reflection_and_rectify_response(question, prediction, label)
        else:
            raise ValueError(f"Error in quality assessment: {is_correct} correct predictions out of {len(predictions)}")