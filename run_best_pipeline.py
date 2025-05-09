#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run the best pipeline for the Multi-Model NLG System.

This script demonstrates the best pipeline for the Multi-Model NLG System,
which combines the best model/system for each task:
- LLaMA model for paraphrase generation
- Ensemble system for question answering
- Pipeline system for summarization
"""

import argparse
import json
import time
from unified_api import UnifiedNLGSystem

def parse_args():
    parser = argparse.ArgumentParser(description="Run the best pipeline for the Multi-Model NLG System")
    parser.add_argument("--input_file", type=str, required=True,
                        help="JSON file containing input data for all tasks")
    parser.add_argument("--output_file", type=str, default="best_pipeline_results.json",
                        help="JSON file to save the results")
    parser.add_argument("--models_dir", type=str, default="fine_tuned_models",
                        help="Directory containing fine-tuned models")
    parser.add_argument("--use_quantization", action="store_true",
                        help="Use quantization for models")
    return parser.parse_args()

def load_input_data(input_file):
    """Load input data from JSON file"""
    with open(input_file, 'r') as f:
        return json.load(f)

def run_pipeline(input_data, models_dir, use_quantization):
    """Run the best pipeline for all tasks"""
    # Initialize the unified system
    system = UnifiedNLGSystem(models_dir, use_quantization)
    
    results = {
        "summarization": [],
        "question_answering": [],
        "paraphrase": []
    }
    
    # Process summarization inputs
    print("\nProcessing summarization inputs...")
    for i, text in enumerate(input_data.get("summarization", [])):
        print(f"  Processing input {i+1}/{len(input_data.get('summarization', []))}")
        start_time = time.time()
        result = system.summarize(text)
        end_time = time.time()
        
        results["summarization"].append({
            "input": text,
            "summary": result["summary"],
            "system_type": result["system_type"],
            "model_name": result.get("model_name"),
            "inference_time": end_time - start_time
        })
    
    # Process question answering inputs
    print("\nProcessing question answering inputs...")
    for i, qa_pair in enumerate(input_data.get("question_answering", [])):
        print(f"  Processing input {i+1}/{len(input_data.get('question_answering', []))}")
        context = qa_pair["context"]
        question = qa_pair["question"]
        
        start_time = time.time()
        result = system.answer_question(context, question)
        end_time = time.time()
        
        results["question_answering"].append({
            "context": context,
            "question": question,
            "answer": result["answer"],
            "system_type": result["system_type"],
            "model_name": result.get("model_name"),
            "inference_time": end_time - start_time
        })
    
    # Process paraphrase inputs
    print("\nProcessing paraphrase inputs...")
    for i, text in enumerate(input_data.get("paraphrase", [])):
        print(f"  Processing input {i+1}/{len(input_data.get('paraphrase', []))}")
        start_time = time.time()
        result = system.generate_paraphrase(text)
        end_time = time.time()
        
        results["paraphrase"].append({
            "input": text,
            "paraphrase": result["paraphrase"],
            "system_type": result["system_type"],
            "model_name": result.get("model_name"),
            "inference_time": end_time - start_time
        })
    
    return results

def save_results(results, output_file):
    """Save results to JSON file"""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")

def print_summary(results):
    """Print summary of results"""
    print("\n" + "="*80)
    print("SUMMARY OF RESULTS")
    print("="*80)
    
    # Summarization summary
    if results["summarization"]:
        print("\nSummarization:")
        avg_time = sum(r["inference_time"] for r in results["summarization"]) / len(results["summarization"])
        print(f"  Processed {len(results['summarization'])} inputs")
        print(f"  Average inference time: {avg_time:.2f} seconds")
        print(f"  System type: {results['summarization'][0]['system_type']}")
    
    # Question answering summary
    if results["question_answering"]:
        print("\nQuestion Answering:")
        avg_time = sum(r["inference_time"] for r in results["question_answering"]) / len(results["question_answering"])
        print(f"  Processed {len(results['question_answering'])} inputs")
        print(f"  Average inference time: {avg_time:.2f} seconds")
        print(f"  System type: {results['question_answering'][0]['system_type']}")
    
    # Paraphrase summary
    if results["paraphrase"]:
        print("\nParaphrase Generation:")
        avg_time = sum(r["inference_time"] for r in results["paraphrase"]) / len(results["paraphrase"])
        print(f"  Processed {len(results['paraphrase'])} inputs")
        print(f"  Average inference time: {avg_time:.2f} seconds")
        print(f"  System type: {results['paraphrase'][0]['system_type']}")
    
    print("\n" + "="*80)

def main():
    args = parse_args()
    
    print("Loading input data...")
    input_data = load_input_data(args.input_file)
    
    print("Running best pipeline...")
    results = run_pipeline(input_data, args.models_dir, args.use_quantization)
    
    print("Saving results...")
    save_results(results, args.output_file)
    
    print_summary(results)
    
    print("\nBest pipeline completed!")

if __name__ == "__main__":
    main()
