import asyncio
import json
import os
from tqdm import tqdm
import utils.data_loader as data_loader
from utils.CoT_generation import CoTGeneration, improve_contrastive_example
import argparse
from utils.evaluate import evaluate  


def main(args):
    ## async generate CoT with concurrency limit
    asyncio.run(improve_contrastive_example(args.dataset_name, args.eg_path, args.model_name))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--eg_path', type=str, required=False)
    parser.add_argument('--concurrency_limit', type=int, default=5, help='concurrency limit of CoT generation (default: 5)')
    args = parser.parse_args()
    
    main(args)





