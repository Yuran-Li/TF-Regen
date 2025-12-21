import asyncio
import json
import os
import random
import numpy as np
from tqdm import tqdm
import utils.data_loader as data_loader
from utils.CoT_generation import CoTGeneration, DATASET_CONFIG
from utils.CoT_assessment import CoTAssessment
import argparse
from utils.evaluate import evaluate

def set_global_seed(seed=42):
    """set global seed to ensure reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    # if torch is installed, set torch seed
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

def convert_format(data):
    """
    change the format of the data from:
    {'field1': [val1, val2], 'field2': [val3, val4]}
    to: [{'field1': val1, 'field2': val3}, {'field1': val2, 'field2': val4}]
    """
    if not data:
        return []
    
    # get all fields
    fields = list(data.keys())
    
    # get data length (assume all fields have the same length)
    num_items = len(data[fields[0]])
    
    # convert format
    converted_data = []
    for i in range(num_items):
        item = {}
        for field in fields:
            item[field] = data[field][i]
        converted_data.append(item)
    
    return converted_data

async def generate_CoT_follow_contrastive(model_name, model_type, dataset_name, data_path, split, method, concurrency_limit=5, seed=42, temperature=0, top_p=1.0):
    set_global_seed(seed)
    
    ##load dataset##
    with open(data_path, "r", encoding="utf-8") as f:
        all_data = json.load(f)
    data = [] ## only process correct questions
    correct_indices = []
    for idx, item in enumerate(all_data):
        raw_assess = item.get("assessment", "")
        if "Assessment:" in raw_assess:
            raw_assess = raw_assess.split("Assessment:")[-1]
        normalized = raw_assess.strip().lower()
        if normalized == "correct":
            data.append(item)
            correct_indices.append(idx)
    # create cot generation instance
    cot_generation = CoTGeneration(model_name=model_name, model_type=model_type, dataset_name=dataset_name, method=method, eg_path=None, seed=seed, temperature=temperature, top_p=top_p)
    
    print(f"processing {len(data)} questions, concurrency limit: {concurrency_limit}")
    
    # manually batch process with progress bar
    cots = []
    total_batches = (len(data) + concurrency_limit - 1) // concurrency_limit
    
    with tqdm(total=len(data), desc="Processing questions", 
              unit="question", unit_scale=True, 
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
        for batch_idx, i in enumerate(range(0, len(data), concurrency_limit), 1):
            batch = data[i:i + concurrency_limit]
            batch_results = await asyncio.gather(*[
                cot_generation.generate_CoT(q) for q in batch
            ])
            cots.extend(batch_results)
            pbar.update(len(batch))
            pbar.set_postfix({"batch": f"{batch_idx}/{total_batches}"})
    
    # prepare results in the required format
    # 仅对被处理（correct）的样本更新 CoT，其余数据原样保留
    for k, idx in enumerate(correct_indices):
        all_data[idx]["CoT"] = cots[k] + [all_data[idx]["CoT"]]  # 用列表拼接
    results = all_data
    
    # create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    
    # save output results
    with open(f"output/{model_name}_{model_type}_{dataset_name}_{split}_CoT_regeneration({method}).json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to output/{model_name}_{model_type}_{dataset_name}_{split}_CoT_regeneration({method}).json")

async def generate_CoT(model_name, model_type, dataset_name, split, method, eg_path, concurrency_limit=5, num_samples=2, seed=42, temperature=0, top_p=1.0):
    set_global_seed(seed)
    
    ##load dataset##
    dataset = data_loader.load_data(dataset_name, split, seed)
    data = convert_format(dataset[:num_samples])
    
    # create cot generation instance
    cot_generation = CoTGeneration(model_name=model_name, model_type=model_type, dataset_name=dataset_name, method=method, eg_path=eg_path, seed=seed, temperature=temperature, top_p=top_p)
    
    print(f"processing {len(data)} questions, concurrency limit: {concurrency_limit}")
    
    # manually batch process with progress bar
    cots = []
    total_batches = (len(data) + concurrency_limit - 1) // concurrency_limit
    
    with tqdm(total=len(data), desc="Processing questions", 
              unit="question", unit_scale=True, 
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
        for batch_idx, i in enumerate(range(0, len(data), concurrency_limit), 1):
            batch = data[i:i + concurrency_limit]
            batch_results = await asyncio.gather(*[
                cot_generation.generate_CoT(q) for q in batch
            ])
            cots.extend(batch_results)
            pbar.update(len(batch))
            pbar.set_postfix({"batch": f"{batch_idx}/{total_batches}"})
    
    # prepare results in the required format
    results = []
    for i, item in enumerate(data):
        item["CoT"] = cots[i]
        results.append(item)
    
    # create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    
    # save output results
    with open(f"output/{model_name}_{model_type}_{dataset_name}_{split}_CoT_generation({method}).json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to output/{model_name}_{model_type}_{dataset_name}_{split}_CoT_generation({method}).json")


async def assess_CoT(model_name, model_type, dataset_name, split, method, generation_path, eg_path, concurrency_limit=5, seed=42):
    set_global_seed(seed)
    
    ##load dataset##
    with open(generation_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # create cot assessment instance
    cot_assess = CoTAssessment(model_name=model_name, model_type=model_type, dataset_name=dataset_name, method=method, eg_path=eg_path, seed=seed)
    
    print(f"processing {len(data)} questions, concurrency limit: {concurrency_limit}")
    
    # manually batch process with progress bar
    assessments = []
    total_batches = (len(data) + concurrency_limit - 1) // concurrency_limit
    
    with tqdm(total=len(data), desc="Processing questions", 
              unit="question", unit_scale=True, 
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
        for batch_idx, i in enumerate(range(0, len(data), concurrency_limit), 1):
            batch = data[i:i + concurrency_limit]
            batch_results = await asyncio.gather(*[
                cot_assess.assess_CoT(q) for q in batch
            ])
            assessments.extend(batch_results)
            pbar.update(len(batch))
            pbar.set_postfix({"batch": f"{batch_idx}/{total_batches}"})
    
    # prepare results in the required format
    results = []
    for i, item in enumerate(data):
        if assessments[i] == "already assessed":
            results.append(item)
        else:
            item["assessment"] = assessments[i]
            results.append(item)
    
    # create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    
    # save output results
    with open(f"output/{model_name}_{model_type}_{dataset_name}_{split}_CoT_assessment({method}).json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to output/{model_name}_{model_type}_{dataset_name}_{split}_CoT_assessment({method}).json")

async def rectify_CoT(model_name, model_type, dataset_name, split, method, assess_path, eg_path, concurrency_limit=5, seed=42, noise_rate=0.0, output_suffix=None):
    set_global_seed(seed)
    
    ##load dataset##
    with open(assess_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    ## only process data with incorrect assessment
    def _normalize_assessment(assess_text: str) -> str:
        if "Assessment:" in assess_text:
            assess_text = assess_text.split("Assessment:")[-1]
        return assess_text.strip().lower()
    # optionally add noise to assessment labels by flipping correct/incorrect
    if noise_rate and noise_rate > 0:
        for item in data:
            norm = _normalize_assessment(item.get("assessment", ""))
            if norm in ("incorrect", "correct"):
                if random.random() < noise_rate:
                    item["assessment"] = "correct" if norm == "incorrect" else "incorrect"
                else:
                    # ensure normalized plain label is stored
                    item["assessment"] = norm
    data_incorrect = [item for item in data if _normalize_assessment(item.get("assessment", "")) == "incorrect"]
    # create cot generation instance
    cot_generation = CoTGeneration(model_name=model_name, model_type=model_type, dataset_name=dataset_name, method=method, eg_path=eg_path, seed=seed)
    
    print(f"processing {len(data_incorrect)} questions, concurrency limit: {concurrency_limit}")
    
    # manually batch process with progress bar
    cots = []
    total_batches = (len(data_incorrect) + concurrency_limit - 1) // concurrency_limit
    
    with tqdm(total=len(data_incorrect), desc="Processing questions", 
              unit="question", unit_scale=True, 
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
        for batch_idx, i in enumerate(range(0, len(data_incorrect), concurrency_limit), 1):
            batch = data_incorrect[i:i + concurrency_limit]
            batch_results = await asyncio.gather(*[
                cot_generation.generate_CoT(q) for q in batch
            ])
            cots.extend(batch_results)
            pbar.update(len(batch))
            pbar.set_postfix({"batch": f"{batch_idx}/{total_batches}"})
    
    # prepare results in the required format
    results = []
    cots_iter = iter(cots)
    for item in data:
        if _normalize_assessment(item.get("assessment", "")) == "incorrect":
            item["rectify_CoT"] = next(cots_iter)
        results.append(item)
    
    # create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    
    # save output results
    rectify_output_path = f"output/{model_name}_{model_type}_{dataset_name}_{split}_CoT_rectify({method})_noise{noise_rate if noise_rate > 0 else '0.0'}{output_suffix if output_suffix else ''}.json"
    with open(rectify_output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to {rectify_output_path}")

    ## evaluate the accuracy
    accuracy, correct, total = evaluate(task=dataset_name, json_file=rectify_output_path)
    
    print(f"accuracy: {accuracy:.2f}%")
    print(f"correct: {correct}")
    print(f"total: {total}")
 
async def main_async(args):
    ##generate
    # await generate_CoT(args.model_name, args.model_type, args.dataset_name, args.split, "zero_shot", args.concurrency_limit, args.num_samples, args.seed)
    # generation_path = f"output/{args.model_name}_{args.model_type}_{args.dataset_name}_{args.split}_CoT_generation(zero_shot).json"
    # accuracy, correct, total = evaluate(task=args.dataset_name, json_file=generation_path)
    # print(f"accuracy: {accuracy:.2f}%")
    # print(f"correct: {correct}")
    # print(f"total: {total}")
    ## assess strategy 1. entropy_based; strategy 2. contrastive_guided/ self_guided; strategy 3. label_based
    assess_method = "label_based"
    # await assess_CoT(args.model_name, args.model_type, args.dataset_name, args.split, assess_method, generation_path, args.eg_path, args.concurrency_limit, args.seed)
    # rectify
    assess_path = f"output/{args.model_name}_{args.model_type}_{args.dataset_name}_{args.split}_CoT_assessment({assess_method}).json"
    # assess_path = f"output/{args.model_name}_{args.model_type}_{args.dataset_name}_{args.split}_CoT_assessment(contrastive+entropy_based).json"
    await rectify_CoT(args.model_name, args.model_type, args.dataset_name, args.split, "contrastive", assess_path, args.eg_path, args.concurrency_limit, args.seed, args.assess_noise_rate)

async def main_iteration_async(args):
    ##generate
    generation_path = f"output/{args.model_name}_{args.model_type}_{args.dataset_name}_{args.split}_CoT_generation(zero_shot).json"
    # ## assess strategy 1. entropy_based; strategy 2. contrastive_guided/ self_guided; strategy 3. label_based
    assess_method = "label_based"
    assess_path = f"output/{args.model_name}_{args.model_type}_{args.dataset_name}_{args.split}_CoT_assessment({assess_method}).json"
    current_generation_path = generation_path
    max_iter = getattr(args, "max_iter", 1)
    rectify_path = None
    for i in range(max_iter):
        ## assess
        await assess_CoT(
            args.model_name, args.model_type, args.dataset_name, args.split,
            assess_method, current_generation_path, args.eg_path,
            args.concurrency_limit, args.seed
        )
        ## rectify
        suffix = f"_iter{i+1}"
        rectify_path = f"output/{args.model_name}_{args.model_type}_{args.dataset_name}_{args.split}_CoT_rectify(contrastive_rectify){suffix}.json"
        await rectify_CoT(
            args.model_name, args.model_type, args.dataset_name, args.split,
            "contrastive_rectify", assess_path, args.eg_path,
            args.concurrency_limit, args.seed, args.assess_noise_rate, output_suffix=suffix
        )
        ## promote rectify_CoT -> CoT for next iteration
        try:
            with open(rectify_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                if "rectify_CoT" in item:
                    item["CoT"] = item["rectify_CoT"]
            with open(rectify_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            current_generation_path = rectify_path
        except FileNotFoundError:
            # if rectify output not found, stop further iterations
            break
    ## evaluate the accuracy
    accuracy, correct, total = evaluate(task=args.dataset_name, json_file=rectify_path if rectify_path else generation_path)
    
    print(f"accuracy: {accuracy:.2f}%")
    print(f"correct: {correct}")
    print(f"total: {total}")

def evaluate_error_detection(data, check_func, method_name):
    config = DATASET_CONFIG[args.dataset_name]
    """评估错误检测的准确率"""
    def _normalize_assessment(assess_text):
        if not isinstance(assess_text, str):
            return ""
        if "Assessment:" in assess_text:
            assess_text = assess_text.split("Assessment:", 1)[-1]
        return assess_text.strip().lower()
    data_incorrect = [item for item in data if _normalize_assessment(item.get("assessment", "")) == "incorrect"]
    data_correct = [item for item in data if _normalize_assessment(item.get("assessment", "")) == "correct"]
    
    # 获取CoT结果，兼容列表和单个元素两种情况
    def get_cot_result(item):
        cot = item["CoT"]
        return cot[-1] if isinstance(cot, list) else cot
    
    TP = len([item for item in data_incorrect if not check_func(get_cot_result(item), item[config['answer_field']])]) ## 准确检测到错误的样本数
    FP = len([item for item in data_incorrect if check_func(get_cot_result(item), item[config['answer_field']])])
    TN = len([item for item in data_correct if check_func(get_cot_result(item), item[config['answer_field']])])
    FN = len([item for item in data_correct if not check_func(get_cot_result(item), item[config['answer_field']])])
    
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0 ## 不误检
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0 ##不漏检
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n{'='*50}")
    print(f"Error Detection Results ({method_name})")
    print(f"{'='*50}")
    print(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    return accuracy, precision, recall, f1

async def evl_error_detection_acc_async(args):
    """评估错误检测准确率"""
    check_func = DATASET_CONFIG[args.dataset_name]["check_func"]
    ### 评估 contrastive_guided ###
    temperature = 0
    top_p = 1
    generation_path = f"output/{args.model_name}_{args.model_type}_{args.dataset_name}_{args.split}_CoT_generation(zero_shot).json"
    
    # assess_method = "contrastive_guided"
    # assess_method = "self_guided"
    # assess_method = "Referi_score"
    # assess_method = "CoT_WP"
    await assess_CoT(args.model_name, args.model_type, args.dataset_name, args.split, assess_method, 
                    generation_path, args.eg_path, args.concurrency_limit, args.seed)
    assess_path = f"output/{args.model_name}_{args.model_type}_{args.dataset_name}_{args.split}_CoT_assessment({assess_method}).json"
    data = json.load(open(assess_path, "r", encoding="utf-8"))
    evaluate_error_detection(data, check_func, assess_method)

    ### 评估 entropy_based ###
    temperature = 0.3
    top_p = 0.7
    # await generate_CoT(args.model_name, args.model_type, args.dataset_name, args.split, "zero_shot", args.eg_path, 
    #                   args.concurrency_limit, args.num_samples, args.seed, temperature, top_p)
    # generation_path = f"output/{args.model_name}_{args.model_type}_{args.dataset_name}_{args.split}_CoT_generation(zero_shot).json"
    
    # assess_method = "entropy_based"
    # await assess_CoT(args.model_name, args.model_type, args.dataset_name, args.split, assess_method, 
    #                 generation_path, args.eg_path, args.concurrency_limit, args.seed)
    # assess_path = f"output/{args.model_name}_{args.model_type}_{args.dataset_name}_{args.split}_CoT_assessment({assess_method}).json"
    # data = json.load(open(assess_path, "r", encoding="utf-8"))
    # evaluate_error_detection(data, check_func, "entropy_based")
    
    ### 评估 2 stage ###
    # generation_path = f"output/{args.model_name}_{args.model_type}_{args.dataset_name}_{args.split}_CoT_generation(zero_shot).json"
    
    # assess_method = "contrastive_guided"
    # await assess_CoT(args.model_name, args.model_type, args.dataset_name, args.split, assess_method, 
    #                 generation_path, args.eg_path, args.concurrency_limit, args.seed)
    # assess_path = f"output/{args.model_name}_{args.model_type}_{args.dataset_name}_{args.split}_CoT_assessment({assess_method}).json"
    # data = json.load(open(assess_path, "r", encoding="utf-8"))
    # evaluate_error_detection(data, check_func, assess_method)

    # temperature = 0.3
    # top_p = 0.7
    # await generate_CoT_follow_contrastive(args.model_name, args.model_type, args.dataset_name, assess_path, args.split, "zero_shot", 
    #                   args.concurrency_limit, args.seed, temperature, top_p)
    # generation_path = f"output/{args.model_name}_{args.model_type}_{args.dataset_name}_{args.split}_CoT_regeneration(zero_shot).json"
    
    # assess_method = "entropy_based"
    # await assess_CoT(args.model_name, args.model_type, args.dataset_name, args.split, assess_method, 
    #                 generation_path, args.eg_path, args.concurrency_limit, args.seed)
    # assess_path = f"output/{args.model_name}_{args.model_type}_{args.dataset_name}_{args.split}_CoT_assessment({assess_method}).json"
    # # assess_path = f"./output/Llama-3.1-8B-Instruct_local_bamboogle_test_CoT_assessment(contrastive_guided).json"
    # data = json.load(open(assess_path, "r", encoding="utf-8"))
    # evaluate_error_detection(data, check_func, "entropy_based")

def main(args):
    # asyncio.run(main_iteration_async(args))
    # asyncio.run(evl_error_detection_acc_async(args))
    asyncio.run(main_async(args))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--split', type=str, required=True)
    parser.add_argument('--eg_path', type=str, required=False)
    parser.add_argument('--concurrency_limit', type=int, default=5, help='concurrency limit of CoT generation (default: 5)')
    parser.add_argument('--num_samples', type=int, default=5, help='number of samples to be processed (default: 5)')
    parser.add_argument('--seed', type=int, default=42, help='random seed for reproducibility (default: 42)')
    parser.add_argument('--max_iter', type=int, default=1, help='max iterations of assess-rectify loop (default: 1)')
    parser.add_argument('--assess_noise_rate', type=float, default=0.0, help='probability to flip assessment correct/incorrect (default: 0.0)')
    args = parser.parse_args()
    
    main(args)





