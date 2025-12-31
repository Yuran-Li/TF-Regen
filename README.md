# code-release

Code for submission.

## Setup

### Requirements
- Python 3.10+
- (Optional, for local inference) NVIDIA GPU (tested on A6000) with CUDA 12.x
- (Optional, for local inference) `vLLM` installed (see `environment.yml`)

### Create and activate the environment
```
conda env create -f environment.yml
conda activate TF-Regen
```

## Reflection Memory Curation
This step constructs the contrastive reflection memory used for later verification and rectification.

### Option 1: Base model via OpenAI API
```
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
bash construct_eg.sh
```

### Option 2: Base model deployed locally (vLLM)
1) Start vLLM server
```
export CUDA_VISIBLE_DEVICES=0,1
export HUGGINGFACE_HUB_TOKEN="YOUR_HF_TOKEN"

python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --served-model-name "Llama-3.1-8B-Instruct" \
  --host 0.0.0.0 --port 8001 \
  --dtype bfloat16 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.70 \
  --max-model-len 8192

```
2) Construct reflection memory
```
bash construct_eg_local.sh
```

## Inputs / Output

### Reflection memory
- `construct_eg*.sh` will generate：
  - `output/{student_model}_{student_type}_{dataset}_{split}_answer_generation(error_pattern).json`
  - `output/{teacher_model}_{teacher_type}_{dataset}_{split}_quality_assessment.json`
- `improve.py` / `improve_contrastive_example.sh` will generate（as `EG_PATH`）：
  - `output/{model_name}_{dataset_name}_improved_eg.json`

### Inference / Regeneration
- `generation.py` generate：
  - `output/{model_name}_{model_type}_{dataset}_{split}_CoT_generation({method}).json`
- `generation_then_correction.py`（verify + rectify）generate：
  - `output/{model_name}_{model_type}_{dataset}_{split}_CoT_assessment({method}).json`
  - `output/{model_name}_{model_type}_{dataset}_{split}_CoT_rectify({method})_noise{rate}.json`

## Inference

### RM-Primed Prompting
```
bash generate_answer.sh
#or (local vLLM)
bash generate_answer_local.sh
```

### RM-Guided Regeneration
This pipeline first verifies the initial answer, then regenerates the response based on the verification results.
```
bash generate_then_rectify_answer.sh
# or (local vLLM)
bash generate_then_rectify_answer_local.sh

```

