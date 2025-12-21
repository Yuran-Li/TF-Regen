##use vllm##
# #!/bin/bash
MODEL_NAME="Llama-3.1-8B-Instruct"
MODEL_TYPE="local"
DATASET_NAME="gsm8k"
NUM_SAMPLES=500
EG_PATH="./output/gpt-5-mini-2025-08-07_legal_improved_eg.json"
SEED=42

export LOCAL_BACKEND=vllm
# 如你的服务不是默认：http://localhost:8000/v1，请设置：
export VLLM_BASE_URL="http://127.0.0.1:8001/v1"

##use hf##
#!/bin/bash
# export HUGGING_FACE_HUB_TOKEN="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"
# MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"  # 或你的本地权重路径
# MODEL_TYPE="local"
# DATASET_NAME="gsm8k"
# NUM_SAMPLES=500
# EG_PATH="./output/gpt-4o-mini_gsm8k_improved_llama-8B_eg.json"
# SEED=42

# export LOCAL_BACKEND=hf
# CUDA_VISIBLE_DEVICES=0
# export HF_DEVICE_MAP=auto
# export HF_DTYPE=bfloat16  # 你的 6000 Ada 支持 bfloat16；如报错可改 float16

# ##contrastive
# python generation.py \
#   --model_name "$MODEL_NAME" \
#   --model_type "$MODEL_TYPE" \
#   --dataset_name "$DATASET_NAME" \
#   --split "test" \
#   --method "contrastive" \
#   --eg_path "$EG_PATH" \
#   --num_samples "$NUM_SAMPLES" \
#   --seed "$SEED" \
#   --concurrency_limit 1

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