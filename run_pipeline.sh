#!/bin/bash

# Set up environment variables
export PYTHONPATH=$(pwd):$PYTHONPATH
export TOKENIZERS_PARALLELISM=false

# Create directories
mkdir -p fine_tuned_models
mkdir -p results

echo "===== Multi-Model NLG System Pipeline ====="

# Step 1: Fine-tune models
echo "===== Step 1: Fine-tuning Models ====="

# Fine-tune Qwen model
echo "Fine-tuning Qwen model..."
python fine_tune_models.py --model qwen --task summarization --num_train_samples 1000 --num_epochs 2 --use_8bit
python fine_tune_models.py --model qwen --task qa --num_train_samples 1000 --num_epochs 2 --use_8bit
python fine_tune_models.py --model qwen --task paraphrase --num_train_samples 1000 --num_epochs 2 --use_8bit

# Fine-tune OPT model
echo "Fine-tuning OPT model..."
python fine_tune_models.py --model opt --task summarization --num_train_samples 1000 --num_epochs 2 --use_8bit
python fine_tune_models.py --model opt --task qa --num_train_samples 1000 --num_epochs 2 --use_8bit
python fine_tune_models.py --model opt --task paraphrase --num_train_samples 1000 --num_epochs 2 --use_8bit

# Fine-tune LLaMA model
echo "Fine-tuning LLaMA model..."
python fine_tune_models.py --model llama --task summarization --num_train_samples 1000 --num_epochs 2 --use_8bit
python fine_tune_models.py --model llama --task qa --num_train_samples 1000 --num_epochs 2 --use_8bit
python fine_tune_models.py --model llama --task paraphrase --num_train_samples 1000 --num_epochs 2 --use_8bit

# Step 2: Evaluate fine-tuned models
echo "===== Step 2: Evaluating Fine-tuned Models ====="

# Evaluate Qwen models
echo "Evaluating Qwen models..."
python evaluate_models.py --model qwen --task summarization --model_path fine_tuned_models/qwen_summarization --output_file results/qwen_summarization.json --use_8bit
python evaluate_models.py --model qwen --task qa --model_path fine_tuned_models/qwen_qa --output_file results/qwen_qa.json --use_8bit
python evaluate_models.py --model qwen --task paraphrase --model_path fine_tuned_models/qwen_paraphrase --output_file results/qwen_paraphrase.json --use_8bit

# Evaluate OPT models
echo "Evaluating OPT models..."
python evaluate_models.py --model opt --task summarization --model_path fine_tuned_models/opt_summarization --output_file results/opt_summarization.json --use_8bit
python evaluate_models.py --model opt --task qa --model_path fine_tuned_models/opt_qa --output_file results/opt_qa.json --use_8bit
python evaluate_models.py --model opt --task paraphrase --model_path fine_tuned_models/opt_paraphrase --output_file results/opt_paraphrase.json --use_8bit

# Evaluate LLaMA models
echo "Evaluating LLaMA models..."
python evaluate_models.py --model llama --task summarization --model_path fine_tuned_models/llama_summarization --output_file results/llama_summarization.json --use_8bit
python evaluate_models.py --model llama --task qa --model_path fine_tuned_models/llama_qa --output_file results/llama_qa.json --use_8bit
python evaluate_models.py --model llama --task paraphrase --model_path fine_tuned_models/llama_paraphrase --output_file results/llama_paraphrase.json --use_8bit

# Step 3: Run multi-model systems
echo "===== Step 3: Running Multi-Model Systems ====="

# Dynamic Decision System
echo "Running Dynamic Decision System..."
python multi_model_system.py --models_dir fine_tuned_models --system_type dynamic --task all --output_file results_dynamic.json --use_quantization

# Ensemble System
echo "Running Ensemble System..."
python multi_model_system.py --models_dir fine_tuned_models --system_type ensemble --task all --output_file results_ensemble.json --use_quantization

# Pipeline System
echo "Running Pipeline System..."
python multi_model_system.py --models_dir fine_tuned_models --system_type pipeline --task all --output_file results_pipeline.json --use_quantization

echo "===== Pipeline Complete ====="
echo "Results are saved in the 'results' directory"
