#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unified API for Multi-Model NLG System

This module provides a simple API to use the best-performing models and systems
for each of the three tasks: summarization, question answering, and paraphrase generation.
"""

import os
import argparse
import logging
import json
import time
from typing import Dict, Any, Optional, Union, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Import project modules
from device_utils import get_device, prepare_model_kwargs
from multi_model_system import (
    ModelManager,
    DynamicDecisionSystem,
    EnsembleSystem,
    PipelineSystem,
    TASK_CONFIGS,
    GENERATION_CONFIG
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("unified_api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class UnifiedNLGSystem:
    """
    Unified API for the Multi-Model NLG System that selects the best model/system for each task.
    """

    def __init__(self, models_dir: str = "fine_tuned_models", use_quantization: bool = False):
        """
        Initialize the unified NLG system.

        Args:
            models_dir: Directory containing fine-tuned models
            use_quantization: Whether to use quantization for models
        """
        self.models_dir = models_dir
        self.use_quantization = use_quantization
        self.model_manager = ModelManager(models_dir, use_quantization)

        # Initialize all systems
        self.dynamic_system = DynamicDecisionSystem(self.model_manager)
        self.ensemble_system = EnsembleSystem(self.model_manager)
        self.pipeline_system = PipelineSystem(self.model_manager)

        # Best system/model for each task based on evaluation results
        self.best_systems = {
            "summarization": ("pipeline", None),  # (system_type, model_name)
            "qa": ("ensemble", None),
            "paraphrase": ("pipeline", None)  # Changed to pipeline system for better paraphrase generation
        }

        # Load task configurations
        self.task_configs = TASK_CONFIGS

    def summarize(self, text: str) -> Dict[str, Any]:
        """
        Generate a summary for the given text.

        Args:
            text: The text to summarize

        Returns:
            Dictionary containing the summary and metadata
        """
        start_time = time.time()

        system_type, model_name = self.best_systems["summarization"]
        task = "summarization"
        prompt_template = self.task_configs[task]["prompt_template"]

        if system_type == "model" and model_name:
            # Use a specific model directly
            model, tokenizer = self.model_manager.load_model(model_name, task)

            # Prepare input
            prompt = prompt_template.format(article=text)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                              max_length=self.task_configs[task]["max_input_length"]).to(model.device)

            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    **GENERATION_CONFIG[task],
                    pad_token_id=tokenizer.pad_token_id
                )

            # Decode
            input_length = inputs.input_ids.shape[1]
            generated_ids = outputs[0, input_length:]
            summary = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            # Unload model
            self.model_manager.unload_model(model_name, task)

        elif system_type == "dynamic":
            # Use dynamic decision system
            summary, _, selected_model = self.dynamic_system.generate(task, text, prompt_template)
            model_name = selected_model

        elif system_type == "ensemble":
            # Use ensemble system
            summary, _ = self.ensemble_system.generate(task, text, prompt_template)

        elif system_type == "pipeline":
            # Use pipeline system
            summary, _ = self.pipeline_system.generate(task, text, prompt_template)

        else:
            raise ValueError(f"Unknown system type: {system_type}")

        end_time = time.time()
        inference_time = end_time - start_time

        return {
            "summary": summary,
            "system_type": system_type,
            "model_name": model_name,
            "inference_time": inference_time
        }

    def answer_question(self, context: str, question: str) -> Dict[str, Any]:
        """
        Answer a question based on the given context.

        Args:
            context: The context text
            question: The question to answer

        Returns:
            Dictionary containing the answer and metadata
        """
        start_time = time.time()

        system_type, model_name = self.best_systems["qa"]
        task = "qa"
        prompt_template = self.task_configs[task]["prompt_template"]
        input_text = {"context": context, "question": question}

        if system_type == "model" and model_name:
            # Use a specific model directly
            model, tokenizer = self.model_manager.load_model(model_name, task)

            # Prepare input
            prompt = prompt_template.format(context=context, question=question)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                              max_length=self.task_configs[task]["max_input_length"]).to(model.device)

            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    **GENERATION_CONFIG[task],
                    pad_token_id=tokenizer.pad_token_id
                )

            # Decode
            input_length = inputs.input_ids.shape[1]
            generated_ids = outputs[0, input_length:]
            answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            # Unload model
            self.model_manager.unload_model(model_name, task)

        elif system_type == "dynamic":
            # Use dynamic decision system
            answer, _, selected_model = self.dynamic_system.generate(task, input_text, prompt_template)
            model_name = selected_model

        elif system_type == "ensemble":
            # Use ensemble system
            answer, _ = self.ensemble_system.generate(task, input_text, prompt_template)

        elif system_type == "pipeline":
            # Use pipeline system
            answer, _ = self.pipeline_system.generate(task, input_text, prompt_template)

        else:
            raise ValueError(f"Unknown system type: {system_type}")

        end_time = time.time()
        inference_time = end_time - start_time

        return {
            "answer": answer,
            "system_type": system_type,
            "model_name": model_name,
            "inference_time": inference_time
        }

    def generate_paraphrase(self, text: str) -> Dict[str, Any]:
        """
        Generate a paraphrase for the given text.

        Args:
            text: The text to paraphrase

        Returns:
            Dictionary containing the paraphrase and metadata
        """
        start_time = time.time()

        system_type, model_name = self.best_systems["paraphrase"]
        task = "paraphrase"
        prompt_template = self.task_configs[task]["prompt_template"]

        if system_type == "model" and model_name:
            # Use a specific model directly
            model, tokenizer = self.model_manager.load_model(model_name, task)

            # Prepare input
            prompt = prompt_template.format(input_question=text)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                              max_length=self.task_configs[task]["max_input_length"]).to(model.device)

            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    **GENERATION_CONFIG[task],
                    pad_token_id=tokenizer.pad_token_id
                )

            # Decode
            input_length = inputs.input_ids.shape[1]
            generated_ids = outputs[0, input_length:]
            paraphrase = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            # Unload model
            self.model_manager.unload_model(model_name, task)

        elif system_type == "dynamic":
            # Use dynamic decision system
            paraphrase, _, selected_model = self.dynamic_system.generate(task, text, prompt_template)
            model_name = selected_model

        elif system_type == "ensemble":
            # Use ensemble system
            paraphrase, _ = self.ensemble_system.generate(task, text, prompt_template)

        elif system_type == "pipeline":
            # Use pipeline system
            paraphrase, _ = self.pipeline_system.generate(task, text, prompt_template)

        else:
            raise ValueError(f"Unknown system type: {system_type}")

        end_time = time.time()
        inference_time = end_time - start_time

        return {
            "paraphrase": paraphrase,
            "system_type": system_type,
            "model_name": model_name,
            "inference_time": inference_time
        }

def parse_args():
    parser = argparse.ArgumentParser(description="Unified API for Multi-Model NLG System")
    parser.add_argument("--task", type=str, required=True, choices=["summarize", "qa", "paraphrase"],
                        help="Task to perform")
    parser.add_argument("--input", type=str, required=True,
                        help="Input text (for summarize and paraphrase) or context (for qa)")
    parser.add_argument("--question", type=str,
                        help="Question (only for qa task)")
    parser.add_argument("--models_dir", type=str, default="fine_tuned_models",
                        help="Directory containing fine-tuned models")
    parser.add_argument("--use_quantization", action="store_true",
                        help="Use quantization for models")
    parser.add_argument("--output_file", type=str,
                        help="File to save the output (optional)")
    return parser.parse_args()

def main():
    args = parse_args()

    # Initialize the unified system
    system = UnifiedNLGSystem(args.models_dir, args.use_quantization)

    # Perform the requested task
    if args.task == "summarize":
        result = system.summarize(args.input)
        print(f"\nSummary: {result['summary']}")

    elif args.task == "qa":
        if not args.question:
            print("Error: --question is required for the qa task")
            return
        result = system.answer_question(args.input, args.question)
        print(f"\nAnswer: {result['answer']}")

    elif args.task == "paraphrase":
        result = system.generate_paraphrase(args.input)
        print(f"\nParaphrase: {result['paraphrase']}")

    # Print metadata
    print(f"\nSystem type: {result['system_type']}")
    if result.get('model_name'):
        print(f"Model name: {result['model_name']}")
    print(f"Inference time: {result['inference_time']:.2f} seconds")

    # Save output to file if requested
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nOutput saved to {args.output_file}")

if __name__ == "__main__":
    main()
