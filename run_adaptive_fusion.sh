#!/bin/bash

# Script to run the Adaptive Fusion system for the Multi-Model NLG System

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

# Parse command line arguments
TASK="all"
NUM_SAMPLES=20
USE_QUANTIZATION=false
LEARNING_RATE=0.1

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --task)
            TASK="$2"
            shift
            shift
            ;;
        --num_samples)
            NUM_SAMPLES="$2"
            shift
            shift
            ;;
        --use_quantization)
            USE_QUANTIZATION=true
            shift
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: ./run_adaptive_fusion.sh [--task summarization|qa|paraphrase|all] [--num_samples N] [--use_quantization] [--learning_rate N]"
            exit 1
            ;;
    esac
done

# Set quantization flag
if [ "$USE_QUANTIZATION" = true ]; then
    QUANT_FLAG="--use_quantization"
else
    QUANT_FLAG=""
fi

echo "===== Running Adaptive Fusion System ====="
echo "Task: $TASK"
echo "Number of samples: $NUM_SAMPLES"
echo "Use quantization: $USE_QUANTIZATION"
echo "Learning rate: $LEARNING_RATE"

# Create directories if they don't exist
mkdir -p results
mkdir -p logs

# Run the Adaptive Fusion system
python src/multi_model_system.py \
    --models_dir fine_tuned_models \
    --system_type adaptive_fusion \
    --task $TASK \
    --num_samples $NUM_SAMPLES \
    --output_file results/adaptive_fusion_${TASK}.json \
    --learning_rate $LEARNING_RATE \
    $QUANT_FLAG

echo "===== Adaptive Fusion System Complete ====="
echo "Results are saved in results/adaptive_fusion_${TASK}.json"
