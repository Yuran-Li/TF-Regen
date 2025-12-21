#!/bin/bash

# Default values
MODEL_NAME="gpt-5-mini-2025-08-07" #gpt-4o-mini #"meta-llama/Llama-3.3-70B-Instruct-Turbo" #meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo, gpt-3.5-turbo-0125
MODEL_TYPE="api"
DATASET_NAME="legal"
NUM_SAMPLES=100
EG_PATH="./output/Headline_gpt-3.5/update/gpt-5-mini-2025-08-07_Headline_improved_eg.json"
SEED=42

# # # run zero shot
python generation.py \
    --model_name "$MODEL_NAME" \
    --model_type "$MODEL_TYPE" \
    --dataset_name "$DATASET_NAME" \
    --split "test" \
    --method "zero_shot" \
    --seed "$SEED" \
    --num_samples "$NUM_SAMPLES" \
    --concurrency_limit 5

## run contrastive
# python generation.py \
#     --model_name "$MODEL_NAME" \
#     --model_type "$MODEL_TYPE" \
#     --dataset_name "$DATASET_NAME" \
#     --split "test" \
#     --method "contrastive" \
#     --eg_path "$EG_PATH" \
#     --num_samples "$NUM_SAMPLES" \
#     --seed "$SEED" \
#     --concurrency_limit 1