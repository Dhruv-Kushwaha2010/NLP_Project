# Multi-Model System for Optimized Natural Language Generation

**Team: Harry Potter and the Goblet of Pretrained Models**
- Dhruv Kushwaha (2021MT10235)
- Tarun Ajay Singh (2021ME10272)

**Course**: ELL884 DEEP LEARNING FOR NATURAL LANGUAGE PROCESSING, Sem-II, 2024-25

This repository contains the code and resources for a multi-model system that leverages the strengths of different pre-trained language models (Qwen2.5-1.5B, OPT-1.3B, and LLaMA-3.2 1B) to optimize performance across multiple natural language generation tasks.

## Project Structure

```
project_structure/
├── code/                      # Source code
│   ├── evaluation/            # Evaluation scripts
│   ├── models/                # Model-specific code
│   ├── pipeline/              # Pipeline implementation
│   └── utils/                 # Utility functions
├── data/                      # Data processing scripts
├── docs/                      # Documentation
├── models/                    # Fine-tuned models
├── presentation/              # Presentation slides
├── report/                    # Project report
└── results/                   # Results and outputs
    ├── figures/               # Visualizations
    ├── logs/                  # Log files
    └── metrics/               # Evaluation metrics
```

## Tasks and Models

We evaluate our system on three key NLG tasks:

1. **Summarization** (CNN/DailyMail dataset) - Qwen2.5-1.5B
2. **Question Answering** (SQuAD 2.0) - OPT-1.3B
3. **Paraphrase Generation** (Quora Question Pairs) - LLaMA-3.2 1B

## Key Features

- **Multi-Model Architecture**: Combines multiple models using different strategies
- **Parameter-Efficient Fine-Tuning**: Uses Low-Rank Adaptation (LoRA)
- **Memory and Inference Optimization**: Optimizes for resource-constrained environments
- **Adaptive Model Fusion**: Dynamically adjusts fusion weights based on input characteristics

## Results

| Model | Task | Metric | Score | Inference Time (s) |
|-------|------|--------|-------|-------------------|
| Qwen | Summarization | ROUGE-L | 0.1658 | 0.9738 |
| OPT | QA | ROUGE-L | 0.0000 | 4.3574 |
| OPT | QA | BERTScore F1 | 0.0000 | 4.3574 |
| LLaMA | Paraphrase | SacreBLEU | 5.0673 | 1.3744 |

## Setup and Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- PEFT 0.4+
- Datasets 2.12+
- Evaluate 0.4+

### Installation

```bash
# Clone the repository
git clone https://github.com/Dhruv-Kushwaha2010/NLP_Project.git
cd NLP_Project

# Create a conda environment
conda create -n nlp_project python=3.10
conda activate nlp_project

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Fine-tuning

```bash
# Fine-tune Qwen on summarization
python code/train_model.py --model qwen --task summarization --dataset_percentage 0.1

# Fine-tune OPT on question answering
python code/train_model.py --model opt --task qa --dataset_percentage 0.1

# Fine-tune LLaMA on paraphrase generation
python code/train_model.py --model llama --task paraphrase --dataset_percentage 0.1
```

### Evaluation

```bash
# Evaluate all models
python code/evaluation/run_all_evaluations.py

# Generate visualizations
python code/evaluation/generate_report_summary.py
```

### Demo

```bash
# Run the demo
python code/demo.py --model qwen --task summarization
```

## Sample Outputs

### Summarization (Qwen)

**Input Article (Excerpt):**
```
Jarryd Hayne's move to the NFL is a boost for rugby league in the United States, it has been claimed. The Australia international full-back or centre quit the National Rugby League in October to try his luck in American football and was this week given a three-year contract with the San Francisco 49ers...
```

**Generated Summary:**
```
Jarryd Hayne, an Australian rugby league player, has signed a three-year contract with the San Francisco 49ers after quitting the National Rugby League. Peter Illfield, chairman of US Association of Rugby League, believes this move will boost rugby league in the United States by creating connections with American football lovers.
```

### Paraphrase Generation (LLaMA)

**Original:** What does it mean when someone has "free domain" over something?
**Paraphrase:** What is the significance when an individual possesses "free domain" regarding an item?

**Original:** How do I increase the fan speed of a cooling pad?
**Paraphrase:** In what way can I enhance the velocity of a cooling pad's fan?

## Limitations and Future Work

- Limited training data (0.1% of datasets)
- Single epoch training
- Output extraction issues
- Model size limitations

Future work should focus on:
- Increasing training data to 5-10% of datasets
- Training for 3-5 epochs with early stopping
- Further refining prompt templates
- Implementing more sophisticated ensemble techniques
- Exploring learned routing networks

## Citation

```
@misc{multi-model-nlg,
  author = {Kushwaha, Dhruv and Singh, Tarun Ajay},
  title = {Multi-Model System for Optimized Natural Language Generation},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Dhruv-Kushwaha2010/NLP_Project}}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

We would like to thank the course instructors and teaching assistants for their guidance and support throughout this project. We also acknowledge the computational resources provided by the department that made this research possible.
