import asyncio
import json
import os
import re

import numpy as np
from tqdm import tqdm
import utils.data_loader as data_loader
from utils.quality import QualityAssessment
from utils.evaluate import evaluate
from utils.CoT_generation import CoTGeneration
import argparse

def extract_numbers_after_hash(text):
    """
    Extract only numbers after "####" from the text
    """
    if "####" in text:
        after_hash = text.split("####")[1].strip()
        return re.sub(r'[^0-9]', '', after_hash)
    return ""

def extract_numbers_only(text):
    """
    Extract only numbers from the text
    """
    return re.sub(r'[^0-9]', '', text.strip())

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

async def construct_eg(model_name, model_type, dataset_name, split, method, concurrency_limit=5, num_samples=2):
    ##load dataset##
    dataset = data_loader.load_data(dataset_name, split)
    data = convert_format(dataset[:num_samples])
    ## create eg construction instance
    cot_generation = CoTGeneration(model_name=model_name, model_type=model_type, dataset_name=dataset_name, method=method, eg_path=None, temperature=0.3, top_p=0.7)
    
    print(f"collecting error patterns of {len(data)} questions, concurrency limit: {concurrency_limit}")
    
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
        item["eg"] = cots[i]
        results.append(item)
    
    # create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    
    # save output results
    file_name = f"output/{model_name}_{model_type}_{dataset_name}_{split}_answer_generation(error_pattern).json"
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {file_name}")

async def analyze_eg(model_name, model_type, dataset_name, split, concurrency_limit=5, original_eg_path=None):
    ##load dataset##
    with open(original_eg_path, "r", encoding="utf-8") as f:
        original_dataset = json.load(f)
    
    data = original_dataset

    # create eg construction instance
    quality_assessment = QualityAssessment(model_name=model_name, model_type=model_type, dataset_name=dataset_name)
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
                quality_assessment.quality_assessment(d) for d in batch
            ])
            assessments.extend(batch_results)
            pbar.update(len(batch))
            pbar.set_postfix({"batch": f"{batch_idx}/{total_batches}"})

    # prepare results in the required format
    results = []
    for i, item in enumerate(data):
        item["assessment"] = assessments[i]
        results.append(item)
    
    # create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    
    # save output results
    file_name = f"output/{model_name}_{model_type}_{dataset_name}_{split}_quality_assessment.json"
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to {file_name}")
    
def main(args):
    ## collect error patterns
    asyncio.run(construct_eg(args.model_student_name, args.model_student_type, args.dataset_name, args.split, "zero_shot", args.concurrency_limit, args.num_samples))
    original_eg_path = f"output/{args.model_student_name}_{args.model_student_type}_{args.dataset_name}_{args.split}_answer_generation(error_pattern).json"
    asyncio.run(analyze_eg(args.model_teacher_name, args.model_teacher_type, args.dataset_name, args.split, args.concurrency_limit, original_eg_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_student_name', type=str, required=True)
    parser.add_argument('--model_student_type', type=str, required=True)
    parser.add_argument('--model_teacher_name', type=str, required=False)
    parser.add_argument('--model_teacher_type', type=str, required=False)
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--split', type=str, required=True)
    parser.add_argument('--method', type=str, required=False)
    parser.add_argument('--concurrency_limit', type=int, default=5, help='concurrency limit of CoT generation (default: 5)')
    parser.add_argument('--num_samples', type=int, default=5, help='number of samples to be processed (default: 5)')
    args = parser.parse_args()
    
    main(args)





