##use vllm##
# #!/bin/bash
MODEL_NAME="Llama-3.1-8B-Instruct"
MODEL_TYPE="local"
DATASET_NAME="gsm8k"
NUM_SAMPLES=500
EG_PATH="./output/gpt-5-mini-2025-08-07_legal_improved_eg.json"
SEED=42

export LOCAL_BACKEND=vllm
export VLLM_BASE_URL="http://127.0.0.1:8001/v1"

# ##zero-shot
  python generation.py \
  --model_name "$MODEL_NAME" \
  --model_type "$MODEL_TYPE" \
  --dataset_name "$DATASET_NAME" \
  --split "test" \
  --method "fixed" \
  --num_samples "$NUM_SAMPLES" \
  --seed "$SEED" \
  --concurrency_limit 5
