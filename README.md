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

### Phase 2: Model Adaptation (Current Phase)
- 🔄 Implement Parameter-Efficient Fine-Tuning (PEFT) using LoRA
- 🔄 Fine-tune each model on each task separately
- 🔄 Evaluate fine-tuned models on validation splits
- 🔄 Compare performance improvements over baseline

### Phase 3: Multi-Model System Design
- Design and implement multi-model architecture:
  - Experiment with dynamic routing mechanisms
  - Develop ensemble techniques for combining model outputs
  - Create pipeline architectures for sequential processing
- Implement efficient inference strategies (caching, batching, etc.)
- Optimize for both performance and computational efficiency

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

## Current Progress
- Successfully completed baseline evaluation for all three models on all three tasks
- Baseline metrics have been calculated and saved
- Environment setup and dependency installation completed
- Ready to begin fine-tuning models using PEFT (LoRA)

## Next Steps
1. Configure and run fine-tuning for each model-task pair
2. Evaluate fine-tuned models and compare with baseline
3. Begin designing multi-model architecture based on fine-tuning results

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

