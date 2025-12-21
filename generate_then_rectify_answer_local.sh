#!/bin/bash
MODEL_NAME="Llama-3.1-8B-Instruct"
MODEL_TYPE="local"
DATASET_NAME="legal"
NUM_SAMPLES=500
EG_PATH="./output/overall_comparison/legal_llama8B/gpt-5-mini-2025-08-07_legal_improved_eg.json"
SEED=42

# NOISE_LIST=${NOISE_LIST:-"0.0 0.2 0.4 0.6 0.8 1.0"}
# REPEATS=${REPEATS:-1}

export LOCAL_BACKEND=vllm
# if not default as：http://localhost:8000/v1，please setup：
export VLLM_BASE_URL="http://127.0.0.1:8001/v1"

# ## run contrastive:
# for NOISE in $NOISE_LIST; do
#     for REP in $(seq 1 $REPEATS); do
#         echo "[RUN] noise=$NOISE repeat=$REP/$REPEATS"
#         python generation_then_correction.py \
#             --model_name "$MODEL_NAME" \
#             --model_type "$MODEL_TYPE" \
#             --dataset_name "$DATASET_NAME" \
#             --split "test" \
#             --eg_path "$EG_PATH" \
#             --num_samples "$NUM_SAMPLES" \
#             --seed "$SEED" \
#             --max_iter 0 \
#             --concurrency_limit 5 \
#             --assess_noise_rate "$NOISE"
#     done
# done


python generation_then_correction.py \
            --model_name "$MODEL_NAME" \
            --model_type "$MODEL_TYPE" \
            --dataset_name "$DATASET_NAME" \
            --split "test" \
            --eg_path "$EG_PATH" \
            --num_samples "$NUM_SAMPLES" \
            --seed "$SEED" \
            --concurrency_limit 5 \
