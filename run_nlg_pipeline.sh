#!/bin/bash

# Streamlined Pipeline Script for Multi-Model NLG System
# This script runs the complete pipeline for the Multi-Model NLG System

# Set up error handling
set -e
echo "Starting Multi-Model NLG System pipeline..."

# Check if we're already in the nlp_project environment
if [[ "$CONDA_DEFAULT_ENV" == "nlp_project" ]]; then
    echo "Already in nlp_project conda environment."
else
    # Try to activate conda environment if available
    if command -v conda &> /dev/null; then
        if conda info --envs | grep -q "nlp_project"; then
            echo "Activating nlp_project conda environment..."
            source $(conda info --base)/etc/profile.d/conda.sh
            conda activate nlp_project
        else
            echo "WARNING: nlp_project conda environment not found. Using current environment."
            echo "You may need to create it first with: conda create -n nlp_project python=3.10 -y"
        fi
    else
        echo "WARNING: conda not found. Using current environment."
    fi
fi

# Set up environment variables
export PYTHONPATH=$(pwd):$PYTHONPATH
export TOKENIZERS_PARALLELISM=false

# Create directories
mkdir -p fine_tuned_models
mkdir -p results

# Parse command line arguments
FULL_PIPELINE=true
USE_QUANTIZATION=false
NUM_TRAIN_SAMPLES=100
NUM_EVAL_SAMPLES=30
NUM_EPOCHS=1

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --quick)
            FULL_PIPELINE=false
            NUM_TRAIN_SAMPLES=20
            NUM_EVAL_SAMPLES=5
            NUM_EPOCHS=1
            shift
            ;;
        --use_quantization)
            USE_QUANTIZATION=true
            shift
            ;;
        --num_train_samples)
            NUM_TRAIN_SAMPLES="$2"
            shift
            shift
            ;;
        --num_eval_samples)
            NUM_EVAL_SAMPLES="$2"
            shift
            shift
            ;;
        --num_epochs)
            NUM_EPOCHS="$2"
            shift
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: ./run_nlg_pipeline.sh [--quick] [--use_quantization] [--num_train_samples N] [--num_eval_samples N] [--num_epochs N]"
            exit 1
            ;;
    esac
done

# Set quantization flag
if [ "$USE_QUANTIZATION" = true ]; then
    QUANT_FLAG="--use_8bit"
else
    QUANT_FLAG=""
fi

echo "===== Multi-Model NLG System Pipeline ====="
echo "Full pipeline: $FULL_PIPELINE"
echo "Use quantization: $USE_QUANTIZATION"
echo "Number of training samples: $NUM_TRAIN_SAMPLES"
echo "Number of evaluation samples: $NUM_EVAL_SAMPLES"
echo "Number of epochs: $NUM_EPOCHS"

# Step 1: Fine-tune models
echo "===== Step 1: Fine-tuning Models ====="

# Fine-tune models for all tasks
echo "Fine-tuning models for all tasks..."

# Summarization
echo "Fine-tuning models for summarization..."
python fine_tune_models.py --model qwen --task summarization --num_train_samples $NUM_TRAIN_SAMPLES --num_epochs $NUM_EPOCHS $QUANT_FLAG
python fine_tune_models.py --model opt --task summarization --num_train_samples $NUM_TRAIN_SAMPLES --num_epochs $NUM_EPOCHS $QUANT_FLAG
python fine_tune_models.py --model llama --task summarization --num_train_samples $NUM_TRAIN_SAMPLES --num_epochs $NUM_EPOCHS $QUANT_FLAG

# Question Answering
echo "Fine-tuning models for question answering..."
python fine_tune_models.py --model qwen --task qa --num_train_samples $NUM_TRAIN_SAMPLES --num_epochs $NUM_EPOCHS $QUANT_FLAG
python fine_tune_models.py --model opt --task qa --num_train_samples $NUM_TRAIN_SAMPLES --num_epochs $NUM_EPOCHS $QUANT_FLAG
python fine_tune_models.py --model llama --task qa --num_train_samples $NUM_TRAIN_SAMPLES --num_epochs $NUM_EPOCHS $QUANT_FLAG

# Paraphrase Generation
echo "Fine-tuning models for paraphrase generation..."
python fine_tune_models.py --model qwen --task paraphrase --num_train_samples $NUM_TRAIN_SAMPLES --num_epochs $NUM_EPOCHS $QUANT_FLAG
python fine_tune_models.py --model opt --task paraphrase --num_train_samples $NUM_TRAIN_SAMPLES --num_epochs $NUM_EPOCHS $QUANT_FLAG
python fine_tune_models.py --model llama --task paraphrase --num_train_samples $NUM_TRAIN_SAMPLES --num_epochs $NUM_EPOCHS $QUANT_FLAG

# Step 2: Evaluate individual models
echo "===== Step 2: Evaluating Individual Models ====="

# Evaluate models for all tasks
echo "Evaluating models for all tasks..."

# Summarization
echo "Evaluating models for summarization..."
python evaluate_models.py --model qwen --task summarization --model_path fine_tuned_models/qwen_summarization --num_samples $NUM_EVAL_SAMPLES --output_file results/qwen_summarization.json $QUANT_FLAG
python evaluate_models.py --model opt --task summarization --model_path fine_tuned_models/opt_summarization --num_samples $NUM_EVAL_SAMPLES --output_file results/opt_summarization.json $QUANT_FLAG
python evaluate_models.py --model llama --task summarization --model_path fine_tuned_models/llama_summarization --num_samples $NUM_EVAL_SAMPLES --output_file results/llama_summarization.json $QUANT_FLAG

# Question Answering
echo "Evaluating models for question answering..."
python evaluate_models.py --model qwen --task qa --model_path fine_tuned_models/qwen_qa --num_samples $NUM_EVAL_SAMPLES --output_file results/qwen_qa.json $QUANT_FLAG
python evaluate_models.py --model opt --task qa --model_path fine_tuned_models/opt_qa --num_samples $NUM_EVAL_SAMPLES --output_file results/opt_qa.json $QUANT_FLAG
python evaluate_models.py --model llama --task qa --model_path fine_tuned_models/llama_qa --num_samples $NUM_EVAL_SAMPLES --output_file results/llama_qa.json $QUANT_FLAG

# Paraphrase Generation
echo "Evaluating models for paraphrase generation..."
python evaluate_models.py --model qwen --task paraphrase --model_path fine_tuned_models/qwen_paraphrase --num_samples $NUM_EVAL_SAMPLES --output_file results/qwen_paraphrase.json $QUANT_FLAG
python evaluate_models.py --model opt --task paraphrase --model_path fine_tuned_models/opt_paraphrase --num_samples $NUM_EVAL_SAMPLES --output_file results/opt_paraphrase.json $QUANT_FLAG
python evaluate_models.py --model llama --task paraphrase --model_path fine_tuned_models/llama_paraphrase --num_samples $NUM_EVAL_SAMPLES --output_file results/llama_paraphrase.json $QUANT_FLAG

# Step 3: Run multi-model systems
echo "===== Step 3: Running Multi-Model Systems ====="

# Run multi-model systems for all tasks
echo "Running multi-model systems for all tasks..."

# Dynamic Decision System
echo "Running Dynamic Decision System..."
python multi_model_system.py --models_dir fine_tuned_models --system_type dynamic --task all --num_samples $NUM_EVAL_SAMPLES --output_file results/dynamic_all.json $QUANT_FLAG

# Ensemble System
echo "Running Ensemble System..."
python multi_model_system.py --models_dir fine_tuned_models --system_type ensemble --task all --num_samples $NUM_EVAL_SAMPLES --output_file results/ensemble_all.json $QUANT_FLAG

# Pipeline System
echo "Running Pipeline System..."
python multi_model_system.py --models_dir fine_tuned_models --system_type pipeline --task all --num_samples $NUM_EVAL_SAMPLES --output_file results/pipeline_all.json $QUANT_FLAG

# Step 4: Compare systems and generate final report
echo "===== Step 4: Comparing Systems and Generating Final Report ====="

echo "Generating final comparison report..."
python compare_systems.py --results_dir results --output_file results/final_comparison.json

# Step 5: Run the best pipeline on sample inputs
echo "===== Step 5: Running Best Pipeline on Sample Inputs ====="

echo "Running best pipeline on sample inputs..."
python run_best_pipeline.py --input_file sample_inputs.json --output_file best_pipeline_results.json $QUANT_FLAG

echo "===== Pipeline Complete ====="
echo "Results are saved in the 'results' directory"
echo "Best pipeline results are saved in 'best_pipeline_results.json'"
