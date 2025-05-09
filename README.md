# Multi-Model System for Optimized Natural Language Generation

## Problem Statement

The goal of this project is to develop a multi-model system that leverages the strengths of different pre-trained models—Qwen2.5-1.5B, OPT-1.3B, and LLaMA-3.2 1B—to optimize performance across multiple tasks in Natural Language Generation (NLG). Unlike traditional single-model systems, this project focuses on combining multiple models in an intelligent and efficient way to balance accuracy, resource usage, and task-specific optimization.

The system uses innovative techniques including:

- **Dynamic Decision Layers**: Decide which model(s) to query based on the input query or task type
- **Pipeline Architectures**: Use one model's output as the input to another, creating a chain of processing for improved results
- **Ensemble Techniques**: Combine predictions from multiple models to produce a superior final output

## Tasks and Evaluation

The system is evaluated on the following tasks and datasets:

1. **Summarization**:
   - Dataset: CNN/DailyMail (news articles → abstractive summaries)
   - Evaluation: ROUGE-L
   - Task: Generate concise and informative summaries of news articles

2. **Question Answering**:
   - Dataset: SQuAD 2.0 (context + question → answer or "no answer")
   - Evaluation: Combination of ROUGE-L and BERTScore
   - Task: Produce free-form answers based on a given context and question

3. **Paraphrase Generation**:
   - Dataset: Quora Question Pairs (questions → paraphrases)
   - Evaluation: Combination of Sacre-BLEU and METEOR
   - Task: Generate semantically equivalent paraphrases for input sentences

## Our Approach

### Multi-Model Architectures

We implemented three different multi-model architectures to optimize performance across tasks:

1. **Dynamic Decision System**:
   - Selects the most appropriate model for each input based on task-specific heuristics
   - For summarization: Selects models based on input length (longer articles → Qwen, shorter articles → LLaMA)
   - For question answering: Selects models based on question complexity and context length
   - For paraphrase generation: Selects models based on input sentence length and complexity

2. **Ensemble System**:
   - Combines predictions from multiple models to produce a superior final output
   - Uses Qwen and OPT models for reliability
   - Implements robust error handling to ensure system stability
   - Includes fallback mechanisms if one model fails

3. **Pipeline System**:
   - Uses specialized prompting techniques with a single model
   - Crafts task-specific prompts to improve output quality
   - Uses Qwen model for all tasks for simplicity and reliability
   - Implements robust error handling to ensure system stability

### Parameter-Efficient Fine-Tuning

All models are fine-tuned using Low-Rank Adaptation (LoRA), a parameter-efficient fine-tuning technique:

- **Memory Efficiency**: LoRA reduces memory requirements by training only a small number of parameters
- **Training Speed**: Fine-tuning is faster compared to full model fine-tuning
- **Performance**: Achieves comparable performance to full fine-tuning with a fraction of the parameters

### Memory and Inference Optimization

Memory usage and inference speed were optimized through several techniques:

- **Model Unloading**: Models are unloaded when no longer needed
- **LRU Caching**: Only the most frequently used models are kept in memory
- **Device Selection**: Automatically selects the fastest available device (CUDA > MPS > CPU)
- **Efficient Prompt Design**: Crafted prompts to minimize the number of tokens generated

## Project Structure

The project is organized into the following directories:

### Source Code (`src/`)

- `device_utils.py`: Utilities for device selection (CUDA > MPS > CPU)
- `load_models.py`: Functions to load the base models
- `load_data.py`: Functions to load and preprocess datasets
- `fine_tune_models.py`: Implementation of LoRA fine-tuning for each model-task pair
- `evaluate_models.py`: Evaluation of model performance using various metrics
- `multi_model_system.py`: Implementation of three multi-model architectures
- `unified_api.py`: Unified API for all three tasks
- `compare_systems.py`: Script to compare different systems and generate reports
- `run_best_pipeline.py`: Script to run the best pipeline for all tasks
- `demo_simplified.py`: Simplified demo script to test the system

### Configuration (`config/`)

- `set_hf_token.py`: Script to set your Hugging Face token for accessing LLaMA model
- `sample_inputs.json`: Sample inputs for testing the system

### Results and Outputs

- `results/`: JSON files with evaluation results
- `plots/`: Visualization plots for model comparisons
- `logs/`: Log files from various components
- `fine_tuned_models/`: Directory containing fine-tuned models

### Scripts

- `run_nlg_pipeline.sh`: Streamlined pipeline script to run the entire system
- `run_demo.sh`: Script to run the demo with various options

## Results and Performance

### Task Performance

| Task | Best System/Model | Metric | Score | Inference Time (s) |
|------|------------------|--------|-------|-------------------|
| Summarization | Pipeline | ROUGE-L | 0.21 | 12.73 |
| Question Answering | Ensemble | BERTScore | 0.84 | 4.50 |
| Paraphrase Generation | LLaMA | SACREBLEU | 22.46 | 3.86 |

### System Comparison

| System | Advantages | Disadvantages |
|--------|------------|---------------|
| Dynamic Decision | - Task-specific optimization<br>- Moderate inference time | - Complex decision logic<br>- Requires all models to be available |
| Ensemble | - Best performance on QA<br>- Robust to model failures | - Slowest inference time<br>- Highest memory usage |
| Pipeline | - Good performance on summarization<br>- Simplified architecture | - Limited to single model capabilities<br>- Less flexible than other systems |

### Key Findings

- LLaMA model performed best on paraphrase generation with a SACREBLEU score of 22.46, significantly outperforming other models
- OPT model achieved a SACREBLEU score of 4.77 for paraphrase generation
- Dynamic decision system achieved a SACREBLEU score of 2.43, outperforming the ensemble system (1.71)
- Ensemble system achieved the best BERTScore (0.84) for QA tasks
- Pipeline system achieved a SACREBLEU score of 3.88 for paraphrase generation, outperforming both dynamic and ensemble systems
- LLaMA model was most efficient with an inference time of 3.86 seconds per sample for paraphrase
- OPT model was slightly slower with an inference time of 4.66 seconds per sample
- Dynamic system took 6.10 seconds per sample, while the ensemble system took 15.90 seconds per sample
- Pipeline system was the most efficient multi-model system with an average inference time of 1.39 seconds per sample for paraphrase

## Setup Instructions

1. **Create Conda Environment**
   ```bash
   conda create -n nlp_project python=3.10 -y
   conda activate nlp_project
   ```

2. **Install Dependencies**
   ```bash
   pip install torch transformers datasets evaluate peft nltk sacrebleu
   ```

3. **Set Hugging Face Token**
   ```bash
   python config/set_hf_token.py YOUR_HUGGINGFACE_TOKEN
   ```

4. **Create Required Directories**
   ```bash
   mkdir -p fine_tuned_models results logs plots data
   ```

## Using the System

### Option 1: Run the Pipeline

```bash
# Run the full pipeline
./run_nlg_pipeline.sh

# Run a quick pipeline (for testing)
./run_nlg_pipeline.sh --quick

# Run with quantization
./run_nlg_pipeline.sh --use_quantization

# Customize the pipeline
./run_nlg_pipeline.sh --num_train_samples 50 --num_eval_samples 20 --num_epochs 2
```

### Option 2: Run the Demo Script

```bash
# Run all tasks with example inputs
./run_demo.sh

# Run a specific task
./run_demo.sh --task summarize
./run_demo.sh --task qa
./run_demo.sh --task paraphrase

# Run with custom input
./run_demo.sh --task summarize --input "Your text to summarize..."
./run_demo.sh --task qa --input "Your context..." --question "Your question...?"
./run_demo.sh --task paraphrase --input "Your text to paraphrase..."

# Run with quantization
./run_demo.sh --use_quantization
```

### Option 3: Use the Unified API in Your Code

```python
from src.unified_api import UnifiedNLGSystem

# Initialize the system
system = UnifiedNLGSystem()

# Summarization
summary = system.summarize("Your text to summarize...")

# Question Answering
answer = system.answer_question("Your context...", "Your question...?")

# Paraphrase Generation
paraphrase = system.generate_paraphrase("Your text to paraphrase...")
```

## Future Improvements

1. **Model Distillation**: Distill knowledge from multiple models into a single, smaller model
2. **Adaptive Batch Processing**: Implement batch processing for multiple inputs to improve throughput
3. **Quantization**: Explore more advanced quantization techniques to reduce memory usage
4. **Prompt Engineering**: Further optimize prompts for each task and model
5. **Hybrid Approaches**: Combine the strengths of different multi-model architectures
6. **Web Interface**: Create a simple web interface for the system
7. **Deployment**: Prepare the system for deployment in a production environment

## References

### Parameter Efficient Fine Tuning
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://arxiv.org/abs/2101.00190)

### Dynamic Decision Layers and Model Routing
- [Mixture of Experts](https://arxiv.org/abs/1701.06538)
- [AdaBERT: Task-Adaptive BERT Compression with Mixture-of-Adapters](https://arxiv.org/abs/2005.04861)

### Ensemble and Modular Techniques
- [Ensemble Methods in Machine Learning](https://link.springer.com/chapter/10.1007/3-540-45014-9_1)
- [RAG: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
