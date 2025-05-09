#!/bin/bash

# Script to run the demo for the Multi-Model NLG System

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
INPUT=""
QUESTION=""
USE_QUANTIZATION=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --task)
            TASK="$2"
            shift
            shift
            ;;
        --input)
            INPUT="$2"
            shift
            shift
            ;;
        --question)
            QUESTION="$2"
            shift
            shift
            ;;
        --use_quantization)
            USE_QUANTIZATION=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: ./run_demo.sh [--task summarize|qa|paraphrase|all] [--input \"text\"] [--question \"question\"] [--use_quantization]"
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

echo "===== Multi-Model NLG System Demo ====="
echo "Task: $TASK"
if [ ! -z "$INPUT" ]; then
    echo "Input: $INPUT"
fi
if [ ! -z "$QUESTION" ]; then
    echo "Question: $QUESTION"
fi
echo "Use quantization: $USE_QUANTIZATION"

# Run the demo
if [ "$TASK" == "all" ]; then
    python src/demo_simplified.py --task all $QUANT_FLAG
elif [ "$TASK" == "summarize" ]; then
    if [ ! -z "$INPUT" ]; then
        python src/demo_simplified.py --task summarize --input "$INPUT" $QUANT_FLAG
    else
        python src/demo_simplified.py --task summarize $QUANT_FLAG
    fi
elif [ "$TASK" == "qa" ]; then
    if [ ! -z "$INPUT" ] && [ ! -z "$QUESTION" ]; then
        python src/demo_simplified.py --task qa --input "$INPUT" --question "$QUESTION" $QUANT_FLAG
    else
        python src/demo_simplified.py --task qa $QUANT_FLAG
    fi
elif [ "$TASK" == "paraphrase" ]; then
    if [ ! -z "$INPUT" ]; then
        python src/demo_simplified.py --task paraphrase --input "$INPUT" $QUANT_FLAG
    else
        python src/demo_simplified.py --task paraphrase $QUANT_FLAG
    fi
else
    echo "Invalid task: $TASK"
    echo "Valid tasks are: summarize, qa, paraphrase, all"
    exit 1
fi

echo "===== Demo Complete ====="
