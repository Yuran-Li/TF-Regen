#!/bin/bash

## Default values
MODEL_student_NAME="Llama-3.1-8B-Instruct" #gpt-3.5-turbo-0125 #gpt-4o-mini #meta-llama/Llama-3.3-70B-Instruct-Turbo, gpt-5-mini-2025-08-07
MODEL_student_TYPE="local"
MODEL_teacher_NAME="gpt-5-mini-2025-08-07" #gpt-4o-mini #meta-llama/Llama-3.3-70B-Instruct-Turbo
MODEL_teacher_TYPE="api"
DATASET_NAME="legal"
NUM_SAMPLES=200

export LOCAL_BACKEND=vllm
# 如你的服务不是默认：http://localhost:8000/v1，请设置：
export VLLM_BASE_URL="http://127.0.0.1:8001/v1"

## construct eg
python construction.py \
    --model_student_name "$MODEL_student_NAME" \
    --model_student_type "$MODEL_student_TYPE" \
    --model_teacher_name "$MODEL_teacher_NAME" \
    --model_teacher_type "$MODEL_teacher_TYPE" \
    --dataset_name "$DATASET_NAME" \
    --split "train" \
    --num_samples "$NUM_SAMPLES" \
    --concurrency_limit 5

# improve eg
EG_PATH="./output/${MODEL_teacher_NAME}_${MODEL_teacher_TYPE}_${DATASET_NAME}_train_quality_assessment.json"
python improve.py \
    --model_name "$MODEL_teacher_NAME" \
    --model_type "$MODEL_teacher_TYPE" \
    --dataset_name "$DATASET_NAME" \
    --eg_path "$EG_PATH" \
    --concurrency_limit 5