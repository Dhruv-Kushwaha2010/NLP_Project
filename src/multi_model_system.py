import os
import torch
import argparse
import logging
import json
import time
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from evaluate import load as load_metric
import nltk
from typing import List, Dict, Any, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("multi_model_system.log"),
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
        "prompt_template": "Based on the context below, answer the question. Provide only the direct answer without any additional text. If the context does not provide the answer, respond with 'unanswerable'.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:",
        "input_columns": ["context", "question"],
        "target_column": "answers",
        "max_input_length": 1024,
        "max_target_length": 128,
        "metrics": ["rouge", "bertscore"]
    },
    "paraphrase": {
        "dataset": "quora",
        "dataset_version": None,
        "prompt_template": "Rewrite this sentence using different words while keeping the same meaning: {input_question}\n\nRewritten:",
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
        "max_new_tokens": 15,       # Reduced max tokens to keep output concise
        "min_new_tokens": 5,        # Ensure at least some output
        "do_sample": False,         # Disable sampling for more deterministic output
        "num_beams": 5,             # Use beam search with more beams for better quality
        "early_stopping": True,     # Stop when a natural ending is reached
        "length_penalty": 0.6       # Prefer shorter outputs
    }
}

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Model System for NLG")
    parser.add_argument("--models_dir", type=str, required=True,
                        help="Directory containing fine-tuned models")
    parser.add_argument("--system_type", type=str, required=True,
                        choices=["dynamic", "ensemble", "pipeline"],
                        help="Type of multi-model system")
    parser.add_argument("--task", type=str, required=True,
                        choices=["summarization", "qa", "paraphrase", "all"],
                        help="Task to evaluate on (or 'all' for all tasks)")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of samples to evaluate")
    parser.add_argument("--output_file", type=str, default="multi_model_results.json",
                        help="File to save evaluation results")
    parser.add_argument("--use_quantization", action="store_true",
                        help="Use 8-bit quantization for all models")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    return parser.parse_args()

class ModelManager:
    def __init__(self, models_dir: str, use_quantization: bool = False, max_cache_size: int = 2):
        self.models_dir = models_dir
        self.use_quantization = use_quantization
        self.loaded_models = {}
        self.loaded_tokenizers = {}
        self.max_cache_size = max_cache_size  # Maximum number of models to keep in memory
        self.model_usage_count = {}  # Track model usage for LRU cache

    def load_model(self, model_name: str, task: str) -> Tuple[Any, Any]:
        """Load a model and its tokenizer for a specific task with LRU caching"""
        model_key = f"{model_name}_{task}"

        # Return cached model if already loaded and update usage count
        if model_key in self.loaded_models:
            # Update usage count for LRU cache
            self.model_usage_count[model_key] = self.model_usage_count.get(model_key, 0) + 1
            logger.info(f"Using cached model: {model_name} for task: {task}")
            return self.loaded_models[model_key], self.loaded_tokenizers[model_key]

        # Check if we need to free up memory before loading a new model
        if len(self.loaded_models) >= self.max_cache_size:
            self._free_least_used_model()

        # Load base model configuration
        model_config = MODEL_CONFIGS[model_name]

        # Load tokenizer
        logger.info(f"Loading tokenizer for {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_config["model_id"],
            **model_config["tokenizer_kwargs"]
        )

        # Ensure tokenizer has pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model
        logger.info(f"Loading model: {model_name} for task: {task}")

        # Import device utilities
        from device_utils import prepare_model_kwargs

        # Prepare model kwargs with device settings
        model_kwargs = prepare_model_kwargs(
            model_config["model_kwargs"].copy(),
            use_quantization=self.use_quantization
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            model_config["model_id"],
            **model_kwargs
        )

        # Load fine-tuned model
        model_path = os.path.join(self.models_dir, f"{model_name}_{task}")
        if os.path.exists(model_path):
            logger.info(f"Loading fine-tuned model from {model_path}")
            model = PeftModel.from_pretrained(base_model, model_path)
        else:
            logger.warning(f"Fine-tuned model not found at {model_path}, using base model")
            model = base_model

        # Cache model and tokenizer
        self.loaded_models[model_key] = model
        self.loaded_tokenizers[model_key] = tokenizer

        # Initialize usage count
        self.model_usage_count[model_key] = 1

        return model, tokenizer

    def _free_least_used_model(self):
        """Free the least recently used model to save memory"""
        if not self.loaded_models:
            return

        # Find the model with the lowest usage count
        least_used_model = min(self.model_usage_count.items(), key=lambda x: x[1])[0]

        logger.info(f"Freeing least used model: {least_used_model} to save memory")

        # Remove the model and tokenizer from cache
        del self.loaded_models[least_used_model]
        del self.loaded_tokenizers[least_used_model]
        del self.model_usage_count[least_used_model]

        # Force garbage collection to free memory
        import gc
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def unload_model(self, model_name: str, task: str):
        """Unload a model to free memory"""
        model_key = f"{model_name}_{task}"
        if model_key in self.loaded_models:
            logger.info(f"Unloading model: {model_name} for task: {task}")
            del self.loaded_models[model_key]
            del self.loaded_tokenizers[model_key]

            # Also remove from usage count
            if model_key in self.model_usage_count:
                del self.model_usage_count[model_key]

            # Force garbage collection
            import gc
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

class DynamicDecisionSystem:
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager

        # Task-specific model preferences based on evaluation performance
        self.task_preferences = {
            "summarization": ["qwen", "llama", "opt"],
            "qa": ["qwen", "llama", "opt"],
            "paraphrase": ["llama", "opt", "qwen"]  # Updated based on SACREBLEU scores
        }

    def select_model(self, task: str, input_text: str) -> str:
        """Select the best model for the given task and input"""

        if task == "summarization":
            # For summarization, use input length to decide
            try:
                input_length = len(input_text.split())
                if input_length > 500:
                    return self.task_preferences[task][0]  # Use best model for long inputs
                else:
                    return self.task_preferences[task][1]  # Use second best for shorter inputs
            except (AttributeError, TypeError):
                # Fallback if input_text is not a string
                logger.warning(f"Unexpected input_text type for summarization: {type(input_text)}")
                return self.task_preferences[task][0]

        elif task == "qa":
            # For QA, input_text is a dictionary with context and question
            if isinstance(input_text, dict) and 'context' in input_text and 'question' in input_text:
                try:
                    context_length = len(input_text['context'].split())
                    question = input_text['question']

                    # Check if the question is complex (contains multiple clauses)
                    is_complex_question = "," in question or ";" in question or len(question.split()) > 20

                    # Check if the context is long
                    is_long_context = context_length > 300

                    if is_complex_question or is_long_context:
                        return self.task_preferences[task][0]  # Use best model for complex questions or long contexts
                    else:
                        return self.task_preferences[task][1]  # Use second best for simple questions with short contexts
                except (AttributeError, TypeError):
                    # Fallback if there's an issue with the input format
                    logger.warning(f"Error processing QA input: {input_text}")
                    return self.task_preferences[task][0]
            else:
                # Fallback if input_text is not in expected format
                logger.warning(f"Unexpected input_text format for QA: {type(input_text)}")
                return self.task_preferences[task][0]

        elif task == "paraphrase":
            # For paraphrase, use input length to decide
            try:
                input_length = len(input_text.split())
                if input_length > 15:
                    return self.task_preferences[task][0]  # Use best model for longer sentences
                else:
                    return self.task_preferences[task][1]  # Use second best for shorter sentences
            except (AttributeError, TypeError):
                # Fallback if input_text is not a string
                logger.warning(f"Unexpected input_text type for paraphrase: {type(input_text)}")
                return self.task_preferences[task][0]

        # Default to the best model for the task
        return self.task_preferences[task][0]

    def generate(self, task: str, input_text: str, prompt_template: str) -> Tuple[str, float, str]:
        """Generate output using the dynamically selected model"""
        selected_model = self.select_model(task, input_text)

        # Load the selected model
        model, tokenizer = self.model_manager.load_model(selected_model, task)

        # Prepare input based on task
        if task == "qa" and isinstance(input_text, dict):
            # For QA, format with context and question
            prompt = prompt_template.format(context=input_text['context'], question=input_text['question'])
        elif task == "summarization":
            # For summarization
            prompt = prompt_template.format(article=input_text)
        elif task == "paraphrase":
            # For paraphrase
            prompt = prompt_template.format(input_question=input_text)
        else:
            # Generic fallback
            prompt = prompt_template.format(**{k: input_text for k in ["article", "context", "question", "input_question"]})

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)

        # Generate
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **GENERATION_CONFIG[task],
                pad_token_id=tokenizer.pad_token_id
            )
        end_time = time.time()
        inference_time = end_time - start_time

        # Decode
        input_length = inputs.input_ids.shape[1]
        generated_ids = outputs[0, input_length:]
        prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        return prediction, inference_time, selected_model

class EnsembleSystem:
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        # Task-specific model selection for better performance
        self.task_models = {
            "summarization": ["qwen"],  # Use only Qwen for summarization (best performance)
            "qa": ["qwen", "opt"],      # Use both for QA
            "paraphrase": ["opt"]       # Use only OPT for paraphrase to avoid LLaMA issues
        }
        # Cache for storing recent predictions
        self.prediction_cache = {}
        self.max_cache_size = 50

    def _get_cached_prediction(self, task: str, input_text: str) -> Tuple[str, float, bool]:
        """Get prediction from cache if available"""
        # Create a cache key based on task and input
        if isinstance(input_text, dict):
            # For QA, create key from context and question
            cache_key = f"{task}_{hash(str(input_text))}"
        else:
            # For other tasks, create key from input text
            cache_key = f"{task}_{hash(input_text)}"

        if cache_key in self.prediction_cache:
            logger.info(f"Using cached prediction for {task}")
            return self.prediction_cache[cache_key]["prediction"], self.prediction_cache[cache_key]["time"], True

        return "", 0.0, False

    def _add_to_cache(self, task: str, input_text: str, prediction: str, inference_time: float):
        """Add prediction to cache"""
        # Create a cache key based on task and input
        if isinstance(input_text, dict):
            # For QA, create key from context and question
            cache_key = f"{task}_{hash(str(input_text))}"
        else:
            # For other tasks, create key from input text
            cache_key = f"{task}_{hash(input_text)}"

        # Add to cache
        self.prediction_cache[cache_key] = {
            "prediction": prediction,
            "time": inference_time
        }

        # Limit cache size
        if len(self.prediction_cache) > self.max_cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.prediction_cache))
            del self.prediction_cache[oldest_key]

    def generate(self, task: str, input_text: str, prompt_template: str) -> Tuple[str, float]:
        """Generate output using ensemble of models - optimized version"""
        # Check cache first
        cached_prediction, cached_time, cache_hit = self._get_cached_prediction(task, input_text)
        if cache_hit:
            return cached_prediction, cached_time

        total_inference_time = 0
        best_prediction = ""

        # Get models for this task
        models = self.task_models.get(task, ["qwen"])  # Default to Qwen if task not found

        # Try each model in order until we get a valid prediction
        for model_name in models:
            try:
                logger.info(f"Trying {model_name} for {task}...")
                model, tokenizer = self.model_manager.load_model(model_name, task)

                # Prepare input based on task
                if task == "qa" and isinstance(input_text, dict):
                    # For QA, format with context and question
                    prompt = prompt_template.format(context=input_text['context'], question=input_text['question'])
                elif task == "summarization":
                    # For summarization
                    prompt = prompt_template.format(article=input_text)
                elif task == "paraphrase":
                    # For paraphrase
                    prompt = prompt_template.format(input_question=input_text)
                else:
                    # Generic fallback
                    prompt = prompt_template.format(**{k: input_text for k in ["article", "context", "question", "input_question"]})

                # Truncate input to improve performance
                max_length = TASK_CONFIGS[task]["max_input_length"]
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(model.device)

                # Generate
                start_time = time.time()
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        **GENERATION_CONFIG[task],
                        pad_token_id=tokenizer.pad_token_id
                    )
                end_time = time.time()
                inference_time = end_time - start_time
                total_inference_time += inference_time

                # Decode
                input_length = inputs.input_ids.shape[1]
                generated_ids = outputs[0, input_length:]
                prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

                # Unload model to free memory
                self.model_manager.unload_model(model_name, task)

                # If we got a valid prediction, use it and stop trying more models
                if prediction:
                    logger.info(f"Got valid prediction from {model_name} for {task}")
                    best_prediction = prediction
                    break
                else:
                    logger.warning(f"Empty prediction from {model_name} for {task}")

            except Exception as e:
                logger.error(f"Error generating with {model_name} for {task}: {e}")
                # Continue with other models

        # Add to cache if we got a valid prediction
        if best_prediction:
            self._add_to_cache(task, input_text, best_prediction, total_inference_time)

        # Return the best prediction we found (or empty string if none)
        return best_prediction, total_inference_time

class PipelineSystem:
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager

        # Use only Qwen for simplicity and reliability
        self.model_name = "qwen"

    def generate(self, task: str, input_text: str, prompt_template: str) -> Tuple[str, float]:
        """Simplified pipeline system that just uses a single model with a specialized prompt"""
        total_inference_time = 0

        try:
            # Load the model
            model, tokenizer = self.model_manager.load_model(self.model_name, task)

            # Create a specialized prompt based on the task
            if task == "qa" and isinstance(input_text, dict):
                # For QA, use a specialized prompt with clear instructions to avoid irrelevant content
                prompt = f"First extract the relevant information, then answer the question. Provide only the direct answer without any additional text or commentary.\n\nContext: {input_text['context']}\n\nQuestion: {input_text['question']}\n\nAnswer:"
            elif task == "summarization":
                # For summarization, use a specialized prompt with clear instructions to avoid irrelevant content
                prompt = f"First identify the key points, then create a concise summary. Focus only on the main points and avoid adding any irrelevant information.\n\nArticle: {input_text}\n\nSummary:"
            elif task == "paraphrase":
                # For paraphrase, use a simpler prompt to avoid verbose outputs
                prompt = f"Rewrite this sentence using different words while keeping the same meaning: {input_text}\n\nRewritten:"
            else:
                # Generic fallback
                prompt = prompt_template.format(**{k: input_text for k in ["article", "context", "question", "input_question"]})

            # Tokenize the input
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)

            # Generate
            start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    **GENERATION_CONFIG[task],
                    pad_token_id=tokenizer.pad_token_id
                )
            end_time = time.time()
            inference_time = end_time - start_time
            total_inference_time += inference_time

            # Decode
            input_length = inputs.input_ids.shape[1]
            generated_ids = outputs[0, input_length:]
            prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            # Unload model to free memory
            self.model_manager.unload_model(self.model_name, task)

            logger.info(f"Pipeline system generated prediction for {task}")
            return prediction, total_inference_time

        except Exception as e:
            logger.error(f"Error in pipeline system for {task}: {e}")
            return "", total_inference_time

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

    # Initialize model manager
    model_manager = ModelManager(args.models_dir, args.use_quantization)

    # Initialize the selected multi-model system
    if args.system_type == "dynamic":
        system = DynamicDecisionSystem(model_manager)
    elif args.system_type == "ensemble":
        system = EnsembleSystem(model_manager)
    elif args.system_type == "pipeline":
        system = PipelineSystem(model_manager)

    # Determine which tasks to evaluate
    tasks = ["summarization", "qa", "paraphrase"] if args.task == "all" else [args.task]

    results = {}

    for task in tasks:
        logger.info(f"Evaluating {args.system_type} system on {task} task")

        # Load task configuration
        task_config = TASK_CONFIGS[task]

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
        if task == "paraphrase":
            dataset = prepare_quora_dataset(dataset, args.num_samples, args.seed)
        else:
            dataset = dataset.shuffle(seed=args.seed).select(range(args.num_samples))

        # Load metrics
        logger.info("Loading evaluation metrics")
        metrics = {}
        for metric_name in task_config["metrics"]:
            metrics[metric_name] = load_metric(metric_name)

        # No need for NLTK data as we're not using METEOR

        # Generate predictions
        logger.info(f"Generating predictions for {len(dataset)} samples")
        predictions = []
        references = []
        total_inference_time = 0
        model_usage_counts = {"qwen": 0, "opt": 0, "llama": 0}

        # Prepare references based on task
        if task == "summarization":
            references = [ex[task_config["target_column"]] for ex in dataset]
            input_texts = [ex[task_config["input_column"]] for ex in dataset]
        elif task == "qa":
            references = [ex["answers"]["text"] if len(ex["answers"]["text"]) > 0 else ["unanswerable"] for ex in dataset]
            input_texts = [{"context": ex["context"], "question": ex["question"]} for ex in dataset]
        elif task == "paraphrase":
            references = [ex[task_config["target_column"]] for ex in dataset]
            input_texts = [ex[task_config["input_column"]] for ex in dataset]

        for i, input_text in enumerate(input_texts):
            try:
                # Generate prediction using the multi-model system
                if args.system_type == "dynamic":
                    # For dynamic system, we need to handle the input differently
                    if task == "qa":
                        # For QA, we need to select based on both context and question
                        # Generate using the selected model
                        prediction, inference_time, selected_model = system.generate(
                            task,
                            input_text,  # Pass the full input_text dictionary
                            task_config["prompt_template"]
                        )
                        model_usage_counts[selected_model] += 1
                    else:
                        prediction, inference_time, selected_model = system.generate(
                            task,
                            input_text,
                            task_config["prompt_template"]
                        )
                        model_usage_counts[selected_model] += 1
                else:
                    # For ensemble and pipeline systems
                    prediction, inference_time = system.generate(
                        task,
                        input_text,
                        task_config["prompt_template"]
                    )

                total_inference_time += inference_time
                predictions.append(prediction)

                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i+1}/{len(input_texts)} samples")

            except Exception as e:
                logger.error(f"Error processing sample {i}: {e}")
                predictions.append("")  # Add empty prediction on error

        # Calculate average inference time
        avg_inference_time = total_inference_time / len(input_texts)
        logger.info(f"Average inference time: {avg_inference_time:.4f} sec/sample")

        # Calculate metrics
        logger.info("Calculating metrics")
        task_results = {
            "system_type": args.system_type,
            "task": task,
            "num_samples": args.num_samples,
            "avg_inference_time": avg_inference_time,
            "metrics": {}
        }

        if args.system_type == "dynamic":
            task_results["model_usage"] = model_usage_counts

        try:
            if "rouge" in metrics and task in ["summarization", "qa"]:
                rouge_results = metrics["rouge"].compute(predictions=predictions, references=references)
                task_results["metrics"]["rouge-l"] = rouge_results["rougeL"]

            if "bertscore" in metrics and task == "qa":
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
                        lang="en"
                    )
                    task_results["metrics"]["bertscore_f1"] = sum(bertscore_results["f1"]) / len(bertscore_results["f1"])

            if "sacrebleu" in metrics and task == "paraphrase":
                # SacreBLEU expects references to be a list of lists
                sacrebleu_references = [[ref] for ref in references]
                sacrebleu_results = metrics["sacrebleu"].compute(predictions=predictions, references=sacrebleu_references)
                task_results["metrics"]["sacrebleu"] = sacrebleu_results["score"]

            if "meteor" in metrics and task == "paraphrase":
                meteor_results = metrics["meteor"].compute(predictions=predictions, references=references)
                task_results["metrics"]["meteor"] = meteor_results["meteor"]

        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            task_results["error"] = str(e)

        results[task] = task_results

        # Print results
        logger.info(f"Results for {task} task:")
        for metric_name, score in task_results["metrics"].items():
            logger.info(f"  {metric_name}: {score:.4f}")

    # Save results
    output_file = args.output_file
    logger.info(f"Saving results to {output_file}")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Evaluation complete!")

if __name__ == "__main__":
    args = parse_args()
    main(args)
