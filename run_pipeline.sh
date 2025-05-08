#!/bin/bash

# Activate conda environment
if command -v conda &> /dev/null; then
    # If conda is available, activate the environment
    if conda info --envs | grep -q "nlp_project"; then
        echo "Activating nlp_project conda environment..."
        source $(conda info --base)/etc/profile.d/conda.sh
        conda activate nlp_project
    else
        echo "WARNING: nlp_project conda environment not found. Using current environment."
    fi
else
    echo "WARNING: conda not found. Using current environment."
fi

# Set up environment variables
export PYTHONPATH=$(pwd):$PYTHONPATH
export TOKENIZERS_PARALLELISM=false

# Create directories
mkdir -p fine_tuned_models
mkdir -p results

echo "===== Multi-Model NLG System Pipeline ====="

# Step 1: Fine-tune models
echo "===== Step 1: Fine-tuning Models ====="

# Optimized fine-tuning approach: Use the best model for each task based on preliminary analysis
echo "Fine-tuning models with optimized approach..."

# For summarization: Qwen performs best
echo "Fine-tuning Qwen for summarization..."
python fine_tune_models.py --model qwen --task summarization --use_8bit

# For QA: LLaMA performs best
echo "Fine-tuning LLaMA for QA..."
python fine_tune_models.py --model llama --task qa --use_8bit

# For paraphrase: OPT performs best
echo "Fine-tuning OPT for paraphrase..."
python fine_tune_models.py --model opt --task paraphrase --use_8bit

# Fine-tune secondary models with even smaller datasets for ensemble/pipeline approaches
echo "Fine-tuning secondary models with reduced dataset..."
python fine_tune_models.py --model opt --task summarization --num_train_samples 50 --use_8bit
python fine_tune_models.py --model qwen --task qa --num_train_samples 50 --use_8bit
python fine_tune_models.py --model llama --task paraphrase --num_train_samples 50 --use_8bit

# Step 2: Evaluate fine-tuned models
echo "===== Step 2: Evaluating Fine-tuned Models ====="

# Evaluate only the models we've fine-tuned
echo "Evaluating fine-tuned models..."

# Primary models (best for each task)
echo "Evaluating primary models..."
python evaluate_models.py --model qwen --task summarization --model_path fine_tuned_models/qwen_summarization --output_file results/qwen_summarization.json --use_8bit --num_samples 50
python evaluate_models.py --model llama --task qa --model_path fine_tuned_models/llama_qa --output_file results/llama_qa.json --use_8bit --num_samples 50
python evaluate_models.py --model opt --task paraphrase --model_path fine_tuned_models/opt_paraphrase --output_file results/opt_paraphrase.json --use_8bit --num_samples 50

# Secondary models (for ensemble/pipeline)
echo "Evaluating secondary models..."
python evaluate_models.py --model opt --task summarization --model_path fine_tuned_models/opt_summarization --output_file results/opt_summarization.json --use_8bit --num_samples 30
python evaluate_models.py --model qwen --task qa --model_path fine_tuned_models/qwen_qa --output_file results/qwen_qa.json --use_8bit --num_samples 30
python evaluate_models.py --model llama --task paraphrase --model_path fine_tuned_models/llama_paraphrase --output_file results/llama_paraphrase.json --use_8bit --num_samples 30

# Step 3: Run multi-model systems
echo "===== Step 3: Running Multi-Model Systems ====="

# Run multi-model systems with reduced sample size for faster evaluation
echo "Running multi-model systems with optimized settings..."

# Dynamic Decision System - Optimized for input-based model selection
echo "Running Dynamic Decision System..."
python multi_model_system.py --models_dir fine_tuned_models --system_type dynamic --task all --output_file results_dynamic.json --use_quantization --num_samples 30

# Ensemble System - Optimized for combining model strengths
echo "Running Ensemble System..."
python multi_model_system.py --models_dir fine_tuned_models --system_type ensemble --task all --output_file results_ensemble.json --use_quantization --num_samples 30

# Pipeline System - Optimized for sequential processing
echo "Running Pipeline System..."
python multi_model_system.py --models_dir fine_tuned_models --system_type pipeline --task all --output_file results_pipeline.json --use_quantization --num_samples 30

# Compare all systems and generate final report
echo "Generating final comparison report..."
python compare_systems.py --results_dir results --output_file final_comparison.json

echo "===== Pipeline Complete ====="
echo "Results are saved in the 'results' directory"
