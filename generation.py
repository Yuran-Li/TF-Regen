import asyncio
import json
import os
import random
import numpy as np
from tqdm import tqdm
import utils.data_loader as data_loader
from utils.CoT_generation import CoTGeneration, improve_contrastive_example
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

async def generate_CoT(model_name, model_type, dataset_name, split, method, eg_path, concurrency_limit=5, num_samples=2, seed=42):
    set_global_seed(seed)
    
    ##load dataset##
    dataset = data_loader.load_data(dataset_name, split, seed)
    data = convert_format(dataset[:num_samples])
    
    # create cot generation instance
    cot_generation = CoTGeneration(model_name=model_name, model_type=model_type, dataset_name=dataset_name, method=method, eg_path=eg_path, seed=seed)
    
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
        


def main(args):
    ## async generate CoT with concurrency limit
    asyncio.run(generate_CoT(args.model_name, args.model_type, args.dataset_name, args.split, args.method, args.eg_path, args.concurrency_limit, args.num_samples, args.seed))
    
    ## evaluate the accuracy
    accuracy, correct, total = evaluate(task=args.dataset_name, json_file=f"output/{args.model_name}_{args.model_type}_{args.dataset_name}_{args.split}_CoT_generation({args.method}).json")
    # accuracy, correct, total = evaluate(task=args.dataset_name, json_file="./output/gsm8k_gpt-3.5/gpt-3.5-turbo-0125_api_gsm8k_test_CoT_generation(contrastive).json")
    
    print(f"accuracy: {accuracy:.2f}%")
    print(f"correct: {correct}")
    print(f"total: {total}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--split', type=str, required=True)
    parser.add_argument('--method', type=str, required=True)
    parser.add_argument('--eg_path', type=str, required=False)
    parser.add_argument('--concurrency_limit', type=int, default=5, help='concurrency limit of CoT generation (default: 5)')
    parser.add_argument('--num_samples', type=int, default=5, help='number of samples to be processed (default: 5)')
    parser.add_argument('--seed', type=int, default=42, help='random seed for reproducibility (default: 42)')
    args = parser.parse_args()
    
    main(args)





