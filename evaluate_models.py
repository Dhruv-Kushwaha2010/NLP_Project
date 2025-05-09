import os
import torch
import argparse
import logging
import time
import json
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from evaluate import load as load_metric
import nltk

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define model configurations
MODEL_CONFIGS = {
    "qwen": {
        "model_id": "Qwen/Qwen2-1.5B",
        "tokenizer_kwargs": {"trust_remote_code": True},
        "model_kwargs": {"trust_remote_code": True},
    },
    "opt": {
        "model_id": "facebook/opt-1.3b",
        "tokenizer_kwargs": {"use_fast": False},
        "model_kwargs": {},
    },
    "llama": {
        "model_id": "meta-llama/Llama-3.2-1B",
        "tokenizer_kwargs": {"token": "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"},
        "model_kwargs": {"token": "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"},
    }
}

# Define task configurations
TASK_CONFIGS = {
    "summarization": {
        "dataset": "cnn_dailymail",
        "dataset_version": "3.0.0",
        "prompt_template": "Summarize the following news article:\n\n{article}\n\nSummary:",
        "input_column": "article",
        "target_column": "highlights",
        "max_input_length": 1024,
        "max_target_length": 256,
        "metrics": ["rouge"]
    },
    "qa": {
        "dataset": "squad_v2",
        "dataset_version": None,
        "prompt_template": "Based on the context below, answer the question. If the context does not provide the answer, respond with 'unanswerable'.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:",
        "input_columns": ["context", "question"],
        "target_column": "answers",
        "max_input_length": 1024,
        "max_target_length": 128,
        "metrics": ["rouge", "bertscore"]
    },
    "paraphrase": {
        "dataset": "quora",
        "dataset_version": None,
        "prompt_template": "Generate a paraphrase for the following sentence:\n\nSentence: {input_question}\n\nParaphrase:",
        "input_column": "input_question",
        "target_column": "reference_paraphrase",
        "max_input_length": 512,
        "max_target_length": 128,
        "metrics": ["sacrebleu"]
    }
}

# Generation parameters
GENERATION_CONFIG = {
    "summarization": {
        "max_new_tokens": 256,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
    },
    "qa": {
        "max_new_tokens": 50,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
    },
    "paraphrase": {
        "max_new_tokens": 100,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
    }
}

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned language models")
    parser.add_argument("--model", type=str, required=True, choices=["qwen", "opt", "llama"],
                        help="Base model to evaluate")
    parser.add_argument("--task", type=str, required=True, choices=["summarization", "qa", "paraphrase"],
                        help="Task to evaluate on")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to fine-tuned model (if None, evaluates base model)")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of samples to evaluate")
    parser.add_argument("--output_file", type=str, default="evaluation_results.json",
                        help="File to save evaluation results")
    parser.add_argument("--use_4bit", action="store_true",
                        help="Use 4-bit quantization")
    parser.add_argument("--use_8bit", action="store_true",
                        help="Use 8-bit quantization")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    return parser.parse_args()

def get_prompt(task_name, example, config):
    if task_name == "summarization":
        return config["prompt_template"].format(article=example[config["input_column"]])
    elif task_name == "qa":
        return config["prompt_template"].format(context=example["context"], question=example["question"])
    elif task_name == "paraphrase":
        return config["prompt_template"].format(input_question=example[config["input_column"]])
    else:
        raise ValueError(f"Unknown task: {task_name}")

def prepare_quora_dataset(dataset, num_samples, seed):
    processed_data = []
    seen_pairs = set()

    # Shuffle indices
    import random
    random.seed(seed)
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    for i in indices:
        pair = dataset[i]['questions']
        pair_id = tuple(sorted((pair['id'][0], pair['id'][1])))

        if pair['id'][0] is not None and pair['id'][1] is not None and pair_id not in seen_pairs:
            processed_data.append({
                'input_question': pair['text'][0],
                'reference_paraphrase': pair['text'][1]
            })
            seen_pairs.add(pair_id)

            if len(processed_data) >= num_samples:
                break

    # Convert to dataset format
    from datasets import Dataset
    return Dataset.from_dict({
        'input_question': [item['input_question'] for item in processed_data],
        'reference_paraphrase': [item['reference_paraphrase'] for item in processed_data]
    })

def main(args):
    torch.manual_seed(args.seed)

    # Load model configuration
    model_config = MODEL_CONFIGS[args.model]
    task_config = TASK_CONFIGS[args.task]

    # Load tokenizer
    logger.info(f"Loading tokenizer for {model_config['model_id']}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["model_id"],
        **model_config["tokenizer_kwargs"]
    )

    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token")

    # Load model
    logger.info(f"Loading model: {model_config['model_id']}")

    # Import device utilities
    from device_utils import get_device, prepare_model_kwargs

    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")

    # Prepare model kwargs with device settings
    model_kwargs = prepare_model_kwargs(
        model_config["model_kwargs"].copy(),
        use_quantization=args.use_8bit,
        device=device
    )

    # Add 4-bit quantization settings if specified (CUDA only)
    if args.use_4bit and device.type == "cuda":
        model_kwargs["load_in_4bit"] = True
        model_kwargs["bnb_4bit_quant_type"] = "nf4"
        model_kwargs["bnb_4bit_compute_dtype"] = torch.bfloat16
        logger.info("Using 4-bit quantization")

    base_model = AutoModelForCausalLM.from_pretrained(
        model_config["model_id"],
        **model_kwargs
    )

    # Load fine-tuned model if specified
    if args.model_path:
        logger.info(f"Loading fine-tuned model from {args.model_path}")
        model = PeftModel.from_pretrained(base_model, args.model_path)
        model_type = "fine-tuned"
    else:
        model = base_model
        model_type = "base"

    # Load dataset
    logger.info(f"Loading dataset: {task_config['dataset']}")

    # Handle specific dataset versions and configurations
    if task_config["dataset"] == "cnn_dailymail":
        logger.info("Loading CNN/DailyMail dataset with version 3.0.0")
        dataset = load_dataset("cnn_dailymail", "3.0.0", split="validation")
    elif task_config["dataset"] == "squad_v2":
        logger.info("Loading SQuAD v2 dataset")
        dataset = load_dataset("squad_v2", split="validation")
    elif task_config["dataset"] == "quora":
        logger.info("Loading Quora Question Pairs dataset (using train split for evaluation)")
        dataset = load_dataset("quora", split="train", trust_remote_code=True)
    else:
        # Fallback to generic loading
        dataset_args = {"split": "validation"}
        if task_config["dataset_version"]:
            dataset_args["version"] = task_config["dataset_version"]
        dataset = load_dataset(task_config["dataset"], **dataset_args)

    # Prepare dataset based on task
    if args.task == "paraphrase":
        dataset = prepare_quora_dataset(dataset, args.num_samples, args.seed)
    else:
        dataset = dataset.shuffle(seed=args.seed).select(range(args.num_samples))

    # Load metrics
    logger.info("Loading evaluation metrics")
    metrics = {}
    for metric_name in task_config["metrics"]:
        metrics[metric_name] = load_metric(metric_name)

    # Ensure NLTK data is available for METEOR
    if "meteor" in metrics:
        try:
            nltk.data.find('corpora/wordnet')
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/omw-1.4')
        except LookupError:
            logger.info("Downloading NLTK data...")
            nltk.download('wordnet', quiet=True)
            nltk.download('punkt', quiet=True)
            nltk.download('omw-1.4', quiet=True)

    # Generate predictions
    logger.info(f"Generating predictions for {len(dataset)} samples")
    predictions = []
    references = []
    total_inference_time = 0

    # Prepare references based on task
    if args.task == "summarization":
        references = [ex[task_config["target_column"]] for ex in dataset]
    elif args.task == "qa":
        references = [ex["answers"]["text"] if len(ex["answers"]["text"]) > 0 else ["unanswerable"] for ex in dataset]
    elif args.task == "paraphrase":
        references = [ex[task_config["target_column"]] for ex in dataset]

    # Get generation config for the task
    gen_config = GENERATION_CONFIG[args.task]

    for i, example in enumerate(dataset):
        prompt = get_prompt(args.task, example, task_config)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=task_config["max_input_length"]).to(model.device)

        # Generate prediction
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **gen_config,
                pad_token_id=tokenizer.pad_token_id
            )
        end_time = time.time()
        total_inference_time += (end_time - start_time)

        # Decode prediction
        input_length = inputs.input_ids.shape[1]
        generated_ids = outputs[0, input_length:]

        try:
            # Handle potential None values in the generated tokens
            prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        except Exception as e:
            logger.warning(f"Error decoding tokens: {e}")
            # Fallback: decode token by token and skip None values
            tokens = []
            for token_id in generated_ids.tolist():
                try:
                    token = tokenizer.convert_ids_to_tokens(token_id)
                    if token is not None:
                        tokens.append(token)
                except:
                    pass

            prediction = tokenizer.convert_tokens_to_string(tokens).strip()

        predictions.append(prediction)

        if (i + 1) % 10 == 0:
            logger.info(f"Processed {i+1}/{len(dataset)} samples")

    # Calculate average inference time
    avg_inference_time = total_inference_time / len(dataset)
    logger.info(f"Average inference time: {avg_inference_time:.4f} sec/sample")

    # Calculate metrics
    logger.info("Calculating metrics")
    results = {
        "model": args.model,
        "model_type": model_type,
        "task": args.task,
        "num_samples": args.num_samples,
        "avg_inference_time": avg_inference_time,
        "metrics": {}
    }

    try:
        if "rouge" in metrics and args.task in ["summarization", "qa"]:
            rouge_results = metrics["rouge"].compute(predictions=predictions, references=references)
            results["metrics"]["rouge-l"] = rouge_results["rougeL"]

        if "bertscore" in metrics and args.task == "qa":
            # Filter out samples with no answers for BERTScore
            filtered_predictions = []
            filtered_references = []

            for pred, ref_list in zip(predictions, references):
                if ref_list:  # If reference list is not empty
                    filtered_predictions.append(pred)
                    filtered_references.append(ref_list)

            if filtered_predictions:
                bertscore_results = metrics["bertscore"].compute(
                    predictions=filtered_predictions,
                    references=filtered_references,
                    lang="en",
                    device=model.device
                )
                results["metrics"]["bertscore_f1"] = sum(bertscore_results["f1"]) / len(bertscore_results["f1"])

        if "sacrebleu" in metrics and args.task == "paraphrase":
            # SacreBLEU expects references to be a list of lists
            sacrebleu_references = [[ref] for ref in references]
            sacrebleu_results = metrics["sacrebleu"].compute(predictions=predictions, references=sacrebleu_references)
            results["metrics"]["sacrebleu"] = sacrebleu_results["score"]

        if "meteor" in metrics and args.task == "paraphrase":
            meteor_results = metrics["meteor"].compute(predictions=predictions, references=references)
            results["metrics"]["meteor"] = meteor_results["meteor"]

    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        results["error"] = str(e)

    # Save results
    logger.info(f"Saving results to {args.output_file}")
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)

    # Print results
    logger.info("Evaluation results:")
    for metric_name, score in results["metrics"].items():
        logger.info(f"  {metric_name}: {score:.4f}")

    logger.info("Evaluation complete!")

if __name__ == "__main__":
    args = parse_args()
    main(args)
