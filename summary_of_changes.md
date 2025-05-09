# Summary of Changes and HPC Fine-tuning Instructions

## Changes Made

1. **Added Dataset Size Information**
   - Updated README.md with dataset sizes:
     - CNN/DailyMail (Summarization): 287,113 samples
     - SQuAD v2 (Question Answering): 130,319 samples
     - Quora Question Pairs (Paraphrase Generation): 404,290 samples

2. **Modified fine_tune_models.py**
   - Added dataset size percentage parameter (--dataset_size_percentage)
   - Added wandb integration (--use_wandb, --wandb_project, --wandb_name)
   - Added display of dataset sizes when --help is called
   - Added calculation of training samples based on percentage if provided

3. **Updated run_nlg_pipeline.sh**
   - Added support for dataset size percentage parameter
   - Added support for wandb parameters
   - Updated fine-tuning commands to use the new parameters

4. **Added Instructions for HPC**
   - Added section in README.md with specific commands for fine-tuning on HPC
   - Added instructions for setting up and using wandb on HPC

## HPC Fine-tuning Commands

To fine-tune the models on HPC with 20% of the dataset, use the following commands:

### 1. Set up the environment

```bash
# Activate the conda environment
conda activate nlp_project

# Set up wandb in offline mode (recommended for HPC)
export WANDB_MODE=offline
export WANDB_API_KEY=your_api_key_here
```

### 2. Fine-tune Qwen model on summarization task

```bash
python src/fine_tune_models.py \
  --model qwen \
  --task summarization \
  --dataset_size_percentage 20 \
  --num_epochs 3 \
  --batch_size 4 \
  --gradient_accumulation_steps 4 \
  --use_8bit \
  --use_wandb \
  --wandb_project "nlp_project_hpc" \
  --wandb_name "qwen_summarization_20pct"
```

### 3. Fine-tune OPT model on question answering task

```bash
python src/fine_tune_models.py \
  --model opt \
  --task qa \
  --dataset_size_percentage 20 \
  --num_epochs 3 \
  --batch_size 4 \
  --gradient_accumulation_steps 4 \
  --use_8bit \
  --use_wandb \
  --wandb_project "nlp_project_hpc" \
  --wandb_name "opt_qa_20pct"
```

### 4. Fine-tune LLaMA model on paraphrase generation task

```bash
python src/fine_tune_models.py \
  --model llama \
  --task paraphrase \
  --dataset_size_percentage 20 \
  --num_epochs 3 \
  --batch_size 4 \
  --gradient_accumulation_steps 4 \
  --use_8bit \
  --use_wandb \
  --wandb_project "nlp_project_hpc" \
  --wandb_name "llama_paraphrase_20pct"
```

### 5. Sync wandb runs when you have internet access

```bash
wandb sync --sync-all
```

## Notes on Dataset Sizes

- Using 20% of the dataset means:
  - Summarization (CNN/DailyMail): ~57,423 samples
  - Question Answering (SQuAD v2): ~26,064 samples
  - Paraphrase Generation (Quora): ~80,858 samples

- These sample sizes are much larger than the default 100 samples, so training will take significantly longer but should produce better models.

- Adjust the batch size, gradient accumulation steps, and other parameters based on the available GPU memory on your HPC system.

- The A100 GPU should be able to handle these settings, but monitor memory usage and adjust if needed.
