#!/bin/bash

# Complete Pipeline Script for Multi-Model NLG System
# This script runs the remaining fine-tuning and evaluation tasks

# Set up error handling
set -e
echo "Starting complete pipeline script..."

# Create directories if they don't exist
mkdir -p fine_tuned_models
mkdir -p results

# 1. Fine-tune models for summarization
echo "Fine-tuning models for summarization..."

# Fine-tune OPT for summarization
echo "Fine-tuning OPT for summarization..."
python fine_tune_models.py --model opt --task summarization --num_train_samples 100 --num_epochs 1

# Fine-tune Qwen for summarization (already started)
echo "Fine-tuning Qwen for summarization..."
python fine_tune_models.py --model qwen --task summarization --num_train_samples 100 --num_epochs 1

# 2. Evaluate individual models
echo "Evaluating individual models..."

# Evaluate OPT on summarization
echo "Evaluating OPT on summarization..."
python evaluate_models.py --model opt --task summarization --model_path fine_tuned_models/opt_summarization --num_samples 5 --output_file results/opt_summarization.json

# Evaluate Qwen on summarization
echo "Evaluating Qwen on summarization..."
python evaluate_models.py --model qwen --task summarization --model_path fine_tuned_models/qwen_summarization --num_samples 5 --output_file results/qwen_summarization.json

# Evaluate OPT on QA
echo "Evaluating OPT on QA..."
python evaluate_models.py --model opt --task qa --model_path fine_tuned_models/opt_qa --num_samples 5 --output_file results/opt_qa.json

# Evaluate Qwen on QA
echo "Evaluating Qwen on QA..."
python evaluate_models.py --model qwen --task qa --model_path fine_tuned_models/qwen_qa --num_samples 5 --output_file results/qwen_qa.json

# Evaluate Qwen on paraphrase
echo "Evaluating Qwen on paraphrase..."
python evaluate_models.py --model qwen --task paraphrase --model_path fine_tuned_models/qwen_paraphrase --num_samples 5 --output_file results/qwen_paraphrase.json

# 3. Evaluate multi-model systems
echo "Evaluating multi-model systems..."

# Evaluate dynamic system on summarization
echo "Evaluating dynamic system on summarization..."
python multi_model_system.py --models_dir fine_tuned_models --system_type dynamic --task summarization --num_samples 5 --output_file results/dynamic_summarization.json

# Evaluate ensemble system on summarization
echo "Evaluating ensemble system on summarization..."
python multi_model_system.py --models_dir fine_tuned_models --system_type ensemble --task summarization --num_samples 5 --output_file results/ensemble_summarization.json

# Evaluate ensemble system on QA
echo "Evaluating ensemble system on QA..."
python multi_model_system.py --models_dir fine_tuned_models --system_type ensemble --task qa --num_samples 5 --output_file results/ensemble_qa.json

# Evaluate pipeline system on all tasks
echo "Evaluating pipeline system on all tasks..."
python multi_model_system.py --models_dir fine_tuned_models --system_type pipeline --task all --num_samples 5 --output_file results/pipeline_all.json

# 4. Generate final comparison
echo "Generating final comparison..."
python compare_systems.py --results_dir results --output_file results/final_comparison.json

echo "Pipeline complete! Check results directory for evaluation results."
