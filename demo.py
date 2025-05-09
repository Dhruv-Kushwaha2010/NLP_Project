#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Demo script for the Unified API of the Multi-Model NLG System.

This script demonstrates how to use the UnifiedNLGSystem for the three tasks:
summarization, question answering, and paraphrase generation.
"""

import argparse
import json
from unified_api import UnifiedNLGSystem

# Example inputs for each task
EXAMPLE_INPUTS = {
    "summarization": """
    The Perseverance rover, NASA's most sophisticated rover to date, landed on Mars on February 18, 2021. 
    The rover's main mission is to search for signs of ancient microbial life on the Red Planet. 
    It will also collect samples of Martian rock and soil that will be returned to Earth in a future mission. 
    Perseverance carries seven scientific instruments, including cameras, spectrometers, and a ground-penetrating radar. 
    It also carries a small helicopter called Ingenuity, which will attempt the first powered flight on another planet. 
    The rover landed in Jezero Crater, a site that scientists believe was once a lake and could contain evidence of past life. 
    The mission is expected to last at least one Martian year, which is about 687 Earth days.
    """,
    
    "qa": {
        "context": """
        The World Wide Web (WWW), commonly known as the Web, is an information system where documents and other web resources are identified by Uniform Resource Locators (URLs), which may be interlinked by hypertext, and are accessible over the Internet. The resources of the WWW are transferred via the Hypertext Transfer Protocol (HTTP) and may be accessed by users by a software application called a web browser and are published by a software application called a web server.
        
        English scientist Tim Berners-Lee invented the World Wide Web in 1989. He wrote the first web browser in 1990 while employed at CERN near Geneva, Switzerland. The browser was released outside CERN in 1991, first to other research institutions starting in January 1991 and to the general public on the Internet in August 1991. The World Wide Web has been central to the development of the Information Age and is the primary tool billions of people use to interact on the Internet.
        """,
        "question": "Who invented the World Wide Web and when?"
    },
    
    "paraphrase": "The quick brown fox jumps over the lazy dog."
}

def parse_args():
    parser = argparse.ArgumentParser(description="Demo for the Unified API of the Multi-Model NLG System")
    parser.add_argument("--task", type=str, choices=["summarize", "qa", "paraphrase", "all"],
                        default="all", help="Task to demonstrate")
    parser.add_argument("--models_dir", type=str, default="fine_tuned_models",
                        help="Directory containing fine-tuned models")
    parser.add_argument("--use_quantization", action="store_true",
                        help="Use quantization for models")
    parser.add_argument("--output_file", type=str,
                        help="File to save the output (optional)")
    return parser.parse_args()

def demo_summarization(system):
    """Demonstrate summarization task"""
    print("\n" + "="*80)
    print("SUMMARIZATION DEMO")
    print("="*80)
    
    text = EXAMPLE_INPUTS["summarization"]
    print("\nInput text:")
    print("-"*80)
    print(text.strip())
    print("-"*80)
    
    print("\nGenerating summary...")
    result = system.summarize(text)
    
    print("\nSummary:")
    print("-"*80)
    print(result["summary"])
    print("-"*80)
    
    print("\nMetadata:")
    print(f"System type: {result['system_type']}")
    if result.get('model_name'):
        print(f"Model name: {result['model_name']}")
    print(f"Inference time: {result['inference_time']:.2f} seconds")
    
    return result

def demo_question_answering(system):
    """Demonstrate question answering task"""
    print("\n" + "="*80)
    print("QUESTION ANSWERING DEMO")
    print("="*80)
    
    context = EXAMPLE_INPUTS["qa"]["context"]
    question = EXAMPLE_INPUTS["qa"]["question"]
    
    print("\nContext:")
    print("-"*80)
    print(context.strip())
    print("-"*80)
    
    print("\nQuestion:")
    print(question)
    
    print("\nGenerating answer...")
    result = system.answer_question(context, question)
    
    print("\nAnswer:")
    print("-"*80)
    print(result["answer"])
    print("-"*80)
    
    print("\nMetadata:")
    print(f"System type: {result['system_type']}")
    if result.get('model_name'):
        print(f"Model name: {result['model_name']}")
    print(f"Inference time: {result['inference_time']:.2f} seconds")
    
    return result

def demo_paraphrase(system):
    """Demonstrate paraphrase generation task"""
    print("\n" + "="*80)
    print("PARAPHRASE GENERATION DEMO")
    print("="*80)
    
    text = EXAMPLE_INPUTS["paraphrase"]
    
    print("\nOriginal text:")
    print(text)
    
    print("\nGenerating paraphrase...")
    result = system.generate_paraphrase(text)
    
    print("\nParaphrase:")
    print("-"*80)
    print(result["paraphrase"])
    print("-"*80)
    
    print("\nMetadata:")
    print(f"System type: {result['system_type']}")
    if result.get('model_name'):
        print(f"Model name: {result['model_name']}")
    print(f"Inference time: {result['inference_time']:.2f} seconds")
    
    return result

def main():
    args = parse_args()
    
    print("Initializing Unified NLG System...")
    system = UnifiedNLGSystem(args.models_dir, args.use_quantization)
    
    results = {}
    
    if args.task == "summarize" or args.task == "all":
        results["summarization"] = demo_summarization(system)
    
    if args.task == "qa" or args.task == "all":
        results["question_answering"] = demo_question_answering(system)
    
    if args.task == "paraphrase" or args.task == "all":
        results["paraphrase"] = demo_paraphrase(system)
    
    # Save output to file if requested
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nOutput saved to {args.output_file}")
    
    print("\nDemo completed!")

if __name__ == "__main__":
    main()
