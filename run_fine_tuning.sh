#!/bin/bash

# Set environment variables
unset BNB_CUDA_VERSION
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/targets/x86_64-linux/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export DISABLE_SLIDING_WINDOW_ATTENTION=true

# Run the fine-tuning script
python src/fine_tune_models.py \
  --model qwen \
  --task summarization \
  --dataset_size_percentage 10 \
  --num_epochs 3 \
  --batch_size 4 \
  --gradient_accumulation_steps 4 \
  --use_8bit \
  --use_wandb \
  --wandb_project "nlp_project_hpc" \
  --wandb_name "qwen_summarization_10pct"
