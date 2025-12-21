#!/bin/bash

# Default values
MODEL_NAME="gpt-5-mini-2025-08-07" #gpt-4o-mini #"meta-llama/Llama-3.3-70B-Instruct-Turbo" #meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo
MODEL_TYPE="api"
DATASET_NAME="math"
EG_PATH="./output/gpt-5-mini-2025-08-07_api_math_train_quality_assessment.json"

# Run the Python script with the provided arguments
python improve.py \
    --model_name "$MODEL_NAME" \
    --model_type "$MODEL_TYPE" \
    --dataset_name "$DATASET_NAME" \
    --eg_path "$EG_PATH" \