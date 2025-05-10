#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to generate a summary of the evaluation results for the report
"""

import os
import json
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("report_summary.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Generate a summary of the evaluation results for the report")
    parser.add_argument("--results_dir", type=str, default="./evaluation_results",
                        help="Directory containing evaluation results")
    parser.add_argument("--output_dir", type=str, default="./report_figures",
                        help="Directory to save report figures")
    
    return parser.parse_args()

def load_results(results_dir):
    """Load all evaluation results from the specified directory"""
    results = {}
    
    # Check if combined results file exists
    combined_file = os.path.join(results_dir, "combined_results.json")
    if os.path.exists(combined_file):
        logger.info(f"Loading combined results from {combined_file}")
        with open(combined_file, 'r') as f:
            return json.load(f)
    
    # Otherwise, load individual result files
    logger.info(f"Loading individual result files from {results_dir}")
    for filename in os.listdir(results_dir):
        if filename.endswith("_results.json") and not filename.startswith("combined"):
            file_path = os.path.join(results_dir, filename)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                model = data.get("model")
                task = data.get("task")
                
                if model and task:
                    if model not in results:
                        results[model] = {}
                    results[model][task] = data
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
    
    return results

def generate_metrics_table(results):
    """Generate a table of metrics for each model and task"""
    headers = ["Model", "Task", "Metric", "Score", "Inference Time (s)"]
    rows = []
    
    for model in results:
        for task in results[model]:
            data = results[model][task]
            metrics = data.get("metrics", {})
            inference_time = data.get("avg_inference_time", "N/A")
            
            for metric_name, score in metrics.items():
                rows.append([model, task, metric_name, f"{score:.4f}", f"{inference_time:.4f}"])
    
    return tabulate(rows, headers=headers, tablefmt="grid")

def plot_metrics_comparison(results, output_dir):
    """Plot metrics comparison for each task"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Group by task
    tasks = {}
    for model in results:
        for task in results[model]:
            if task not in tasks:
                tasks[task] = {}
            tasks[task][model] = results[model][task]
    
    # Plot for each task
    for task in tasks:
        metrics = set()
        for model in tasks[task]:
            metrics.update(tasks[task][model].get("metrics", {}).keys())
        
        for metric in metrics:
            plt.figure(figsize=(10, 6))
            
            models = []
            scores = []
            
            for model in tasks[task]:
                if metric in tasks[task][model].get("metrics", {}):
                    models.append(model)
                    scores.append(tasks[task][model]["metrics"][metric])
            
            plt.bar(models, scores)
            plt.title(f"{task.capitalize()} - {metric}")
            plt.ylabel("Score")
            plt.ylim(0, max(scores) * 1.2)  # Add some space above the highest bar
            
            # Add values on top of bars
            for i, score in enumerate(scores):
                plt.text(i, score + (max(scores) * 0.05), f"{score:.4f}", 
                         ha='center', va='bottom', fontweight='bold')
            
            # Save figure
            filename = f"{task}_{metric}.png"
            plt.savefig(os.path.join(output_dir, filename))
            plt.close()
            
            logger.info(f"Saved figure: {filename}")

def plot_inference_time_comparison(results, output_dir):
    """Plot inference time comparison for all models and tasks"""
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    
    # Prepare data
    models = list(results.keys())
    tasks = set()
    for model in results:
        tasks.update(results[model].keys())
    tasks = list(tasks)
    
    # Set up plot
    x = np.arange(len(models))
    width = 0.8 / len(tasks)
    
    # Plot bars for each task
    for i, task in enumerate(tasks):
        times = []
        for model in models:
            if task in results[model]:
                times.append(results[model][task].get("avg_inference_time", 0))
            else:
                times.append(0)
        
        plt.bar(x + i * width - width * (len(tasks) - 1) / 2, times, width, label=task)
    
    plt.xlabel('Models')
    plt.ylabel('Inference Time (seconds)')
    plt.title('Average Inference Time by Model and Task')
    plt.xticks(x, models)
    plt.legend()
    
    # Save figure
    filename = "inference_time_comparison.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    
    logger.info(f"Saved figure: {filename}")

def generate_latex_table(results):
    """Generate LaTeX table for the report"""
    latex_table = "\\begin{table}[h]\n\\centering\n\\begin{tabular}{|l|l|l|r|r|}\n\\hline\n"
    latex_table += "Model & Task & Metric & Score & Inference Time (s) \\\\ \\hline\n"
    
    for model in sorted(results.keys()):
        for task in sorted(results[model].keys()):
            data = results[model][task]
            metrics = data.get("metrics", {})
            inference_time = data.get("avg_inference_time", "N/A")
            
            for metric_name in sorted(metrics.keys()):
                score = metrics[metric_name]
                latex_table += f"{model} & {task} & {metric_name} & {score:.4f} & {inference_time:.4f} \\\\ \\hline\n"
    
    latex_table += "\\end{tabular}\n\\caption{Evaluation results for different models and tasks}\n\\label{tab:results}\n\\end{table}"
    
    return latex_table

def main():
    args = parse_args()
    
    # Load results
    results = load_results(args.results_dir)
    
    if not results:
        logger.error(f"No results found in {args.results_dir}")
        return 1
    
    # Generate metrics table
    logger.info("Generating metrics table")
    metrics_table = generate_metrics_table(results)
    print("\nMetrics Table:")
    print(metrics_table)
    
    # Generate LaTeX table
    logger.info("Generating LaTeX table")
    latex_table = generate_latex_table(results)
    print("\nLaTeX Table:")
    print(latex_table)
    
    # Plot metrics comparison
    logger.info("Plotting metrics comparison")
    plot_metrics_comparison(results, args.output_dir)
    
    # Plot inference time comparison
    logger.info("Plotting inference time comparison")
    plot_inference_time_comparison(results, args.output_dir)
    
    logger.info(f"Report summary generated. Figures saved to {args.output_dir}")
    
    return 0

if __name__ == "__main__":
    main()
