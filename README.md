# Multi-Model System for Optimized Natural Language Generation

## Problem Statement
The goal of this project is to develop a multi-model system that leverages the strengths of different pre-trained models—Qwen2.5-1.5B, OPT-1.3B, and LLaMA-3.2 1B—to optimize performance across multiple tasks in Natural Language Generation (NLG). Unlike traditional single-model systems, this project focuses on combining multiple models in an intelligent and efficient way to balance accuracy, resource usage, and task-specific optimization.

Students are encouraged to design systems that use innovative techniques, including but not limited to:

- **Dynamic Decision Layers**: Decide which model(s) to query based on the input query or task type.
- **Pipeline Architectures**: Use one model's output as the input to another, creating a chain of processing for improved results.
- **Ensemble Techniques**: Combine predictions from multiple models to produce a superior final output.

The challenge lies in creating an efficient system that achieves high performance across tasks while minimizing redundancy and computational cost.

## Tasks and Datasets
The system will be evaluated on the following tasks and datasets:

1. **Summarization**:
   - Dataset: CNN/DailyMail (news articles → abstractive summaries).
   - Task: Generate concise and informative summaries of news articles.
2. **Question Answering**:
   - Dataset: SQuAD 2.0 (context + question → answer or "no answer").
   - Task: Produce free-form answers based on a given context and question.
3. **Paraphrase Generation**:
   - Dataset: Quora Question Pairs (questions → paraphrases).
   - Task: Generate semantically equivalent paraphrases for input sentences.

Only the train split of these datasets is allowed for training purposes. The test split will be used for leaderboard evaluation.

- [CNN/DailyMail Dataset](https://huggingface.co/datasets/cnn_dailymail)
- [SQuAD 2.0 Dataset](https://huggingface.co/datasets/squad)
- [Quora Question Pairs Dataset](https://huggingface.co/datasets/quora)

## Evaluation Metrics
The quality of the generated outputs will be assessed using the following metrics:

- **Summarization**: ROUGE-L.
- **Question Answering**: Combination of ROUGE-L and BERTScore.
- **Paraphrase Generation**: Combination of Sacre-BLEU and METEOR.
- **Efficiency**: Inference time per query.

The final leaderboard score will combine all these metrics, evaluated on the test splits of the specified datasets.

## Model Constraints
- Only allowed pre-trained language models:
  - Qwen2.5-1.5B
  - OPT-1.3B
  - LLaMA-3.2 1B
- Fine-tuning on the train splits of the specified datasets is allowed.
- Publicly available models explicitly fine-tuned for these tasks are not allowed.

## Project Plan

### Phase 1: Baseline Evaluation (Completed)
- ✅ Set up environment and dependencies
- ✅ Load base models (Qwen, OPT, LLaMA)
- ✅ Implement zero-shot inference for all three tasks
- ✅ Calculate baseline metrics (ROUGE-L, BERTScore, SacreBLEU, METEOR)
- ✅ Analyze baseline performance

### Phase 2: Model Adaptation (Partially Completed)
- ✅ Implement Parameter-Efficient Fine-Tuning (PEFT) using LoRA
- 🔄 Fine-tune each model on each task separately
  - ✅ OPT for paraphrase generation
  - ✅ OPT for question answering
  - ✅ Qwen for paraphrase generation
  - ✅ Qwen for question answering
  - 🔄 Qwen for summarization (started)
  - ❌ OPT for summarization
  - ❌ LLaMA for all tasks (authentication issues)
- 🔄 Evaluate fine-tuned models on validation splits
  - ✅ OPT on paraphrase generation
  - ✅ Qwen on paraphrase generation
  - ✅ Qwen on question answering
  - ❌ Remaining model-task pairs
- 🔄 Compare performance improvements over baseline

### Phase 3: Multi-Model System Design (Partially Completed)
- ✅ Design and implement multi-model architecture:
  - ✅ Experiment with dynamic routing mechanisms
  - ✅ Develop ensemble techniques for combining model outputs
  - 🔄 Create pipeline architectures for sequential processing
- 🔄 Implement efficient inference strategies (caching, batching, etc.)
- 🔄 Optimize for both performance and computational efficiency
- 🔄 Evaluate multi-model systems:
  - ✅ Dynamic system on paraphrase generation
  - ✅ Ensemble system on paraphrase generation
  - ❌ Remaining system-task pairs

### Phase 4: System Integration and Optimization
- Integrate all components into a unified system
- Implement task-specific optimizations
- Develop a unified API for all three tasks
- Optimize inference time and resource usage
- Conduct ablation studies to identify the most effective components

### Phase 5: Evaluation and Refinement
- Comprehensive evaluation on validation splits
- Error analysis and system refinement
- Prepare for final leaderboard submission
- Document system architecture and design decisions
- Prepare final report and presentation

## Current Progress (Last Updated: May 9, 2025)
- Successfully completed baseline evaluation for all three models on all three tasks
- Baseline metrics have been calculated and saved
- Environment setup and dependency installation completed
- Implemented PEFT (LoRA) fine-tuning for all models
- Created multi-model architectures (Dynamic Decision, Ensemble, Pipeline)
- Implemented device-aware loading (CUDA > MPS > CPU)
- Added optimization techniques for efficient inference
- Fine-tuned OPT and Qwen models for paraphrase generation and question answering
- Started fine-tuning Qwen for summarization
- Evaluated individual models and multi-model systems on paraphrase generation
- Implemented comparison tools for analyzing performance across models and systems

### Key Findings
- OPT model performed best on paraphrase generation with a SACREBLEU score of 4.77
- Dynamic decision system achieved a SACREBLEU score of 2.43, outperforming the ensemble system (1.71)
- OPT model was the fastest with an inference time of 4.66 seconds per sample
- Dynamic system took 6.10 seconds per sample, while the ensemble system took 15.90 seconds per sample

## Implementation Details

### Project Structure
- `device_utils.py`: Utilities for device selection (CUDA > MPS > CPU)
- `load_models.py`: Functions to load the base models
- `load_data.py`: Functions to load and preprocess datasets
- `fine_tune_models.py`: Implementation of LoRA fine-tuning for each model-task pair
- `evaluate_models.py`: Evaluation of model performance using various metrics
- `multi_model_system.py`: Implementation of three multi-model architectures
- `compare_systems.py`: Comparison of different multi-model architectures
- `run_pipeline.sh`: Shell script to run the entire pipeline
- `set_hf_token.py`: Script to set your Hugging Face token for accessing LLaMA model

### Setup Instructions

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
   python set_hf_token.py YOUR_HUGGINGFACE_TOKEN
   ```

### Running the Pipeline

**Option 1: Run the Full Pipeline**
```bash
bash run_pipeline.sh
```

**Option 2: Run the Remaining Tasks Pipeline**
```bash
bash complete_pipeline.sh
```
This script will run the remaining fine-tuning and evaluation tasks that haven't been completed yet.

**Option 3: Run Individual Steps**

Fine-tune a Model:
```bash
python fine_tune_models.py --model [qwen|opt|llama] --task [summarization|qa|paraphrase] --use_8bit
```

Evaluate a Model:
```bash
python evaluate_models.py --model [qwen|opt|llama] --task [summarization|qa|paraphrase] --model_path fine_tuned_models/[model]_[task] --output_file results/[model]_[task].json --use_8bit
```

Run a Multi-Model System:
```bash
python multi_model_system.py --models_dir fine_tuned_models --system_type [dynamic|ensemble|pipeline] --task [summarization|qa|paraphrase|all] --output_file results/[system_type]_results.json --use_quantization
```

Compare Systems:
```bash
python compare_systems.py --results_dir results --output_file final_comparison.json
```

## Next Steps
1. **Complete Fine-tuning**:
   - Resolve LLaMA model access issues by properly setting up the Hugging Face token
   - Complete fine-tuning for summarization (OPT and Qwen models)
   - Run `complete_pipeline.sh` to automate the remaining fine-tuning tasks

2. **Fix Multi-Model Systems**:
   - Update the dynamic decision and ensemble systems to handle the QA task correctly
   - Implement and evaluate the pipeline system for all tasks

3. **Comprehensive Evaluation**:
   - Evaluate all models and systems on all three tasks
   - Generate comparison plots and tables for all tasks
   - Run the `compare_systems.py` script to analyze performance across all systems

4. **Optimization**:
   - Implement memory optimization techniques
   - Improve inference speed for ensemble and pipeline systems
   - Optimize the best-performing architecture for each task

5. **Prepare for Final Submission**:
   - Document system architecture and design decisions
   - Prepare final report and presentation
   - Create a unified API for all three tasks

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

