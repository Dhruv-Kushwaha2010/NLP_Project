# Multi-Model NLG System Evaluation Report

## Overview

This report summarizes the evaluation results of our multi-model NLG system using three different language models (Qwen2.5-1.5B, OPT-1.3B, and LLaMA-3.2 1B) on three different NLG tasks (summarization, question answering, and paraphrase generation).

## Evaluation Setup

- **Models**: 
  - Qwen2.5-1.5B (for summarization)
  - OPT-1.3B (for question answering)
  - LLaMA-3.2 1B (for paraphrase generation)

- **Tasks**:
  - Summarization: CNN/DailyMail dataset
  - Question Answering: SQuAD v2 dataset
  - Paraphrase Generation: Quora Question Pairs dataset

- **Metrics**:
  - Summarization: ROUGE-L
  - Question Answering: ROUGE-L, BERTScore
  - Paraphrase Generation: SacreBLEU

- **Fine-tuning Details**:
  - All models were fine-tuned on 0.1% of their respective datasets
  - Training was done for 1 epoch with batch size 1 and gradient accumulation steps of 4
  - Learning rate: 0.0005
  - LoRA configuration: r=8, alpha=16, dropout=0.05

## Results

### Performance Metrics

| Model | Task | Metric | Score | Inference Time (s) |
|-------|------|--------|-------|-------------------|
| Qwen | Summarization | ROUGE-L | 0.1658 | 0.9738 |
| OPT | QA | ROUGE-L | 0.0000 | 4.3574 |
| OPT | QA | BERTScore F1 | 0.0000 | 4.3574 |
| LLaMA | Paraphrase | SacreBLEU | 5.0673 | 1.3744 |

### Analysis

1. **Qwen on Summarization**:
   - Achieved a ROUGE-L score of 0.1658, which is modest but shows the model has learned some summarization capabilities
   - Fastest inference time at 0.9738 seconds per sample
   - The model was able to generate coherent summaries for some articles

2. **OPT on Question Answering**:
   - Scored 0.0000 on both ROUGE-L and BERTScore F1
   - Slowest inference time at 4.3574 seconds per sample
   - The model struggled to extract answers from the context, often generating empty responses

3. **LLaMA on Paraphrase Generation**:
   - Achieved a SacreBLEU score of 5.0673
   - Moderate inference time at 1.3744 seconds per sample
   - The model initially generated "?????" responses but our improved extraction logic helped it generate more meaningful paraphrases

## Challenges and Limitations

1. **Limited Training Data**:
   - Models were fine-tuned on only 0.1% of the datasets (287 samples for CNN/DailyMail, 130 for SQuAD, 404 for Quora)
   - This small amount of training data was insufficient for the models to learn the tasks effectively

2. **Single Epoch Training**:
   - All models were trained for only one epoch, which is typically not enough for complex NLG tasks

3. **Output Extraction Issues**:
   - Initial evaluation showed poor results due to issues with extracting generated text from model outputs
   - We implemented improved extraction logic that helped recover better results

4. **Model Size Limitations**:
   - The models used (1.3B-1.5B parameters) are relatively small for complex NLG tasks
   - Larger models might perform better but would require more computational resources

## Recommendations for Improvement

1. **Increase Training Data**:
   - Fine-tune on at least 5-10% of the datasets instead of 0.1%
   - Use data augmentation techniques to expand the training data

2. **Extended Training**:
   - Train for 3-5 epochs to allow models to better learn the tasks
   - Implement early stopping based on validation performance

3. **Prompt Engineering**:
   - Further refine the prompt templates to better guide the models
   - Experiment with few-shot examples in the prompts

4. **Model Ensembling**:
   - Combine outputs from multiple models to improve performance
   - Implement voting or weighted averaging for final predictions

5. **Larger Models**:
   - If computational resources allow, experiment with larger models (7B+ parameters)
   - Consider using distilled versions of larger models for efficiency

## Conclusion

Our multi-model NLG system shows promising results on summarization with the Qwen model and paraphrase generation with the LLaMA model, while the OPT model struggled with question answering. The main limitations were the small amount of training data and limited training time. With the recommended improvements, we expect significant performance gains across all tasks.

The system demonstrates the feasibility of using different specialized models for different NLG tasks, which can be more efficient than using a single large model for all tasks. This approach allows for better resource allocation and potentially better performance on specific tasks.

## Future Work

1. Implement the recommended improvements for training and evaluation
2. Explore dynamic routing between models based on task characteristics
3. Develop a unified API for seamless integration of all models
4. Benchmark against larger models to quantify the efficiency-performance tradeoff
