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
        "model_id": "meta-llama/Meta-Llama-3.1-8B",
        "tokenizer_kwargs": {},
        "model_kwargs": {},
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
        "metrics": ["sacrebleu", "meteor"]
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
    def __init__(self, models_dir: str, use_quantization: bool = False):
        self.models_dir = models_dir
        self.use_quantization = use_quantization
        self.loaded_models = {}
        self.loaded_tokenizers = {}
    
    def load_model(self, model_name: str, task: str) -> Tuple[Any, Any]:
        """Load a model and its tokenizer for a specific task"""
        model_key = f"{model_name}_{task}"
        
        # Return cached model if already loaded
        if model_key in self.loaded_models:
            return self.loaded_models[model_key], self.loaded_tokenizers[model_key]
        
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
        model_kwargs = model_config["model_kwargs"].copy()
        model_kwargs["device_map"] = "auto"
        
        if self.use_quantization:
            model_kwargs["load_in_8bit"] = True
        else:
            model_kwargs["torch_dtype"] = torch.bfloat16
        
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
        
        return model, tokenizer
    
    def unload_model(self, model_name: str, task: str):
        """Unload a model to free memory"""
        model_key = f"{model_name}_{task}"
        if model_key in self.loaded_models:
            logger.info(f"Unloading model: {model_name} for task: {task}")
            del self.loaded_models[model_key]
            del self.loaded_tokenizers[model_key]
            torch.cuda.empty_cache()

class DynamicDecisionSystem:
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        
        # Task-specific model preferences based on baseline performance
        self.task_preferences = {
            "summarization": ["qwen", "llama", "opt"],
            "qa": ["qwen", "llama", "opt"],
            "paraphrase": ["qwen", "opt", "llama"]
        }
    
    def select_model(self, task: str, input_text: str) -> str:
        """Select the best model for the given task and input"""
        # Simple heuristic: use input length to decide
        input_length = len(input_text.split())
        
        if task == "summarization":
            if input_length > 500:
                return self.task_preferences[task][0]  # Use best model for long inputs
            else:
                return self.task_preferences[task][1]  # Use second best for shorter inputs
        elif task == "qa":
            # For QA, check if the question is complex (contains multiple clauses)
            if "," in input_text or ";" in input_text or input_length > 20:
                return self.task_preferences[task][0]  # Use best model for complex questions
            else:
                return self.task_preferences[task][1]  # Use second best for simple questions
        elif task == "paraphrase":
            if input_length > 15:
                return self.task_preferences[task][0]  # Use best model for longer sentences
            else:
                return self.task_preferences[task][1]  # Use second best for shorter sentences
        
        # Default to the best model for the task
        return self.task_preferences[task][0]
    
    def generate(self, task: str, input_text: str, prompt_template: str) -> Tuple[str, float, str]:
        """Generate output using the dynamically selected model"""
        selected_model = self.select_model(task, input_text)
        
        # Load the selected model
        model, tokenizer = self.model_manager.load_model(selected_model, task)
        
        # Prepare input
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
        self.models = ["qwen", "opt", "llama"]
        
        # Model weights based on baseline performance
        self.model_weights = {
            "summarization": {"qwen": 0.5, "opt": 0.2, "llama": 0.3},
            "qa": {"qwen": 0.6, "opt": 0.1, "llama": 0.3},
            "paraphrase": {"qwen": 0.5, "opt": 0.3, "llama": 0.2}
        }
    
    def generate(self, task: str, input_text: str, prompt_template: str) -> Tuple[str, float]:
        """Generate output using ensemble of models"""
        predictions = []
        total_inference_time = 0
        
        # Get predictions from all models
        for model_name in self.models:
            model, tokenizer = self.model_manager.load_model(model_name, task)
            
            # Prepare input
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
            total_inference_time += inference_time
            
            # Decode
            input_length = inputs.input_ids.shape[1]
            generated_ids = outputs[0, input_length:]
            prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            
            predictions.append((prediction, model_name))
            
            # Unload model to free memory
            self.model_manager.unload_model(model_name, task)
        
        # For summarization and QA, use weighted voting
        if task in ["summarization", "qa"]:
            # Simple weighted combination (could be improved with more sophisticated methods)
            final_prediction = self._weighted_combination(predictions, task)
        # For paraphrase, use the best model's prediction
        else:
            # Sort by model weight and take the best
            sorted_predictions = sorted(predictions, key=lambda x: self.model_weights[task][x[1]], reverse=True)
            final_prediction = sorted_predictions[0][0]
        
        return final_prediction, total_inference_time
    
    def _weighted_combination(self, predictions: List[Tuple[str, str]], task: str) -> str:
        """Combine predictions using weighted voting"""
        # This is a simple implementation - could be improved with more sophisticated methods
        weighted_predictions = [(pred, self.model_weights[task][model]) for pred, model in predictions]
        
        # For now, just return the prediction with the highest weight
        # In a real system, you might want to use more sophisticated combination methods
        sorted_predictions = sorted(weighted_predictions, key=lambda x: x[1], reverse=True)
        return sorted_predictions[0][0]

class PipelineSystem:
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        
        # Define pipeline configurations for each task
        self.pipelines = {
            "summarization": [
                ("qwen", self._extract_key_points),  # First model extracts key points
                ("llama", self._generate_final_summary)  # Second model generates final summary
            ],
            "qa": [
                ("qwen", self._extract_relevant_context),  # First model extracts relevant context
                ("llama", self._generate_answer)  # Second model generates answer
            ],
            "paraphrase": [
                ("qwen", self._generate_paraphrase)  # Single model for paraphrase
            ]
        }
    
    def generate(self, task: str, input_text: str, prompt_template: str) -> Tuple[str, float]:
        """Generate output using a pipeline of models"""
        total_inference_time = 0
        current_text = input_text
        
        for i, (model_name, processor_func) in enumerate(self.pipelines[task]):
            model, tokenizer = self.model_manager.load_model(model_name, task)
            
            # Prepare custom prompt based on pipeline stage
            if i == 0:
                # First stage uses the original prompt template
                prompt = prompt_template.format(**{k: current_text for k in ["article", "context", "question", "input_question"]})
            else:
                # Later stages use custom prompts defined in processor functions
                prompt = processor_func(current_text)
            
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
            
            # Update current_text for next stage in pipeline
            current_text = prediction
            
            # Unload model to free memory
            self.model_manager.unload_model(model_name, task)
        
        return current_text, total_inference_time
    
    # Pipeline stage processor functions
    def _extract_key_points(self, text: str) -> str:
        return f"Extract the key points from this article:\n\n{text}\n\nKey points:"
    
    def _generate_final_summary(self, key_points: str) -> str:
        return f"Generate a coherent summary based on these key points:\n\n{key_points}\n\nSummary:"
    
    def _extract_relevant_context(self, text: str) -> str:
        return f"Extract the most relevant information from this context to answer the question:\n\n{text}\n\nRelevant information:"
    
    def _generate_answer(self, relevant_context: str) -> str:
        return f"Based on this information, answer the question:\n\n{relevant_context}\n\nAnswer:"
    
    def _generate_paraphrase(self, text: str) -> str:
        # Single-stage pipeline for paraphrase
        return f"Generate a paraphrase for this sentence:\n\n{text}\n\nParaphrase:"

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
                        # Combine context and question for dynamic selection
                        combined_text = f"Context: {input_text['context']}\nQuestion: {input_text['question']}"
                        prediction, inference_time, selected_model = system.generate(
                            task, 
                            input_text, 
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
    output_file = f"{args.system_type}_{args.output_file}"
    logger.info(f"Saving results to {output_file}")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info("Evaluation complete!")

if __name__ == "__main__":
    args = parse_args()
    main(args)
