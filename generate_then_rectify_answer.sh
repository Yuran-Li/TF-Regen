#!/bin/bash

# Default values
MODEL_NAME="gpt-3.5-turbo-0125" #gpt-4o-mini #"meta-llama/Llama-3.3-70B-Instruct-Turbo" #meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo, gpt-3.5-turbo-0125
MODEL_TYPE="api"
DATASET_NAME="gsm_hard"
NUM_SAMPLES=500
EG_PATH="./output/overall_comparison/gsm_hard_gpt-3.5/gpt-5-mini-2025-08-07_gsm_hard_improved_eg(rewrite).json"
SEED=42

## run contrastive
python generation_then_correction.py \
    --model_name "$MODEL_NAME" \
    --model_type "$MODEL_TYPE" \
    --dataset_name "$DATASET_NAME" \
    --split "test" \
    --eg_path "$EG_PATH" \
    --num_samples "$NUM_SAMPLES" \
    --seed "$SEED" \
    --concurrency_limit 5