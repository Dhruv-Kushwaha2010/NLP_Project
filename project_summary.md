# Multi-Model NLG System Project Summary

## Project Overview
This project implements a multi-model system for Natural Language Generation (NLG) tasks using three pre-trained models:
- Qwen2.5-1.5B
- OPT-1.3B
- LLaMA-3.2 1B

The system is designed to handle three NLG tasks:
1. Summarization (CNN/DailyMail dataset)
2. Question Answering (SQuAD 2.0 dataset)
3. Paraphrase Generation (Quora Question Pairs dataset)

## Implemented Architectures
We've implemented three multi-model architectures:

1. **Dynamic Decision System**: Selects the most appropriate model for each input based on input characteristics.
2. **Ensemble System**: Combines outputs from multiple models to produce a final result.
3. **Pipeline System**: Uses models in sequence, with one model's output feeding into another.

## Current Progress

### Fine-tuning Status
| Model | Summarization | Question Answering | Paraphrase Generation |
|-------|---------------|-------------------|----------------------|
| Qwen  | Started       | ✅ Completed      | ✅ Completed         |
| OPT   | Not Started   | ✅ Completed      | ✅ Completed         |
| LLaMA | Not Started   | Not Started       | Not Started          |

### Evaluation Results

#### Paraphrase Generation
| System/Model    | SACREBLEU          | Inference Time (s)   |
|-----------------|--------------------|--------------------|
| OPT             | 4.77               | 4.66               |
| Dynamic System  | 2.43               | 6.10               |
| Ensemble System | 1.71               | 15.90              |

The OPT model performed best on the paraphrase generation task, achieving the highest SACREBLEU score while also being the fastest in terms of inference time. The dynamic decision system performed better than the ensemble system but was slightly slower than the individual OPT model.

#### Question Answering
Evaluation for the QA task is still in progress. Initial results show that the fine-tuned models are working, but there are issues with the multi-model system implementation for this task.

#### Summarization
Evaluation for the summarization task has not yet been completed.

## Challenges and Solutions

### Hugging Face Token Authentication
We encountered issues with accessing the LLaMA model due to authentication problems. This requires setting up the Hugging Face token correctly using the `set_hf_token.py` script and ensuring the token is properly configured in the model loading code.

### Multi-Model System Implementation
The dynamic decision system works well for paraphrase generation but has issues with the QA task. This appears to be due to differences in the data format between tasks. We need to update the multi-model system to handle the specific format of each task's dataset.

### Hardware Constraints
We're running on MPS (Metal Performance Shaders) which doesn't support 8-bit quantization. This limits our ability to optimize memory usage and may impact performance.

## Next Steps

1. **Complete Fine-tuning**:
   - Resolve LLaMA model access issues
   - Fine-tune all models for summarization

2. **Fix Multi-Model Systems**:
   - Update the dynamic decision and ensemble systems to handle QA task correctly
   - Implement and evaluate the pipeline system

3. **Comprehensive Evaluation**:
   - Evaluate all models and systems on all three tasks
   - Generate comparison plots and tables for all tasks

4. **Optimization**:
   - Implement memory optimization techniques
   - Improve inference speed for ensemble and pipeline systems

5. **Final Report**:
   - Complete analysis of all systems
   - Provide recommendations for optimal multi-model architecture

## Conclusion
The project has made significant progress in implementing and evaluating multi-model architectures for NLG tasks. The OPT model has shown strong performance on paraphrase generation, and the dynamic decision system shows promise as a multi-model architecture. Further work is needed to complete the evaluation on all tasks and optimize the system for production use.
