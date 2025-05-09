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
        logging.FileHandler("comparison.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Compare multi-model system results")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory containing result files")
    parser.add_argument("--output_file", type=str, default="final_comparison.json",
                        help="Output file for comparison results")
    return parser.parse_args()

def load_results(results_dir):
    """Load all result files from the directory"""
    results = {}

    # Load all JSON files in the directory
    for filename in os.listdir(results_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(results_dir, filename)
            with open(file_path, 'r') as f:
                try:
                    data = json.load(f)

                    # Check if it's a model result file
                    if 'model' in data and 'task' in data:
                        model = data['model']
                        task = data['task']
                        key = f"{model}_{task}"
                        results[key] = data
                        logger.info(f"Loaded results for {model} on {task}")

                    # Check if it's a system result file (dynamic, ensemble, pipeline)
                    elif any(system_type in filename for system_type in ["dynamic", "ensemble", "pipeline"]):
                        system_type = None
                        for st in ["dynamic", "ensemble", "pipeline"]:
                            if st in filename:
                                system_type = st
                                break

                        if system_type:
                            task = None
                            for t in ["summarization", "qa", "paraphrase"]:
                                if t in filename or t in str(data):
                                    task = t
                                    break

                            if task:
                                key = f"system_{system_type}_{task}"
                                results[key] = data
                                logger.info(f"Loaded results for {system_type} system on {task}")
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse JSON from {file_path}")

    return results

def calculate_metrics(results):
    """Calculate comparison metrics across models and systems"""
    metrics = {}

    # Define tasks and their metrics
    task_metrics = {
        "summarization": ["rouge_l"],
        "qa": ["rouge_l", "bertscore_f1"],
        "paraphrase": ["sacrebleu", "meteor"]
    }

    # Process individual model results
    for model_task, result in results.items():
        if model_task.startswith("system_"):
            continue

        model, task = model_task.split("_")
        if task in task_metrics:
            # Check if metrics are in the top level or in a 'metrics' field
            metrics_data = result.get('metrics', result)

            for metric_name in task_metrics[task]:
                if metric_name in metrics_data:
                    key = f"{model}_{task}_{metric_name}"
                    metrics[key] = metrics_data[metric_name]

                    # Also store inference time
                    if "avg_inference_time" in result:
                        metrics[f"{model}_{task}_time"] = result["avg_inference_time"]

    # Process system results
    for system_key, system_result in results.items():
        if not system_key.startswith("system_"):
            continue

        system_type, task = system_key.split("_")[1:]

        # Check if the result has a task field or is directly the task data
        task_result = system_result.get(task, system_result)

        # Check if metrics are in the top level or in a 'metrics' field
        metrics_data = task_result.get('metrics', task_result)

        for metric_name in task_metrics.get(task, []):
            if metric_name in metrics_data:
                key = f"system_{system_type}_{task}_{metric_name}"
                metrics[key] = metrics_data[metric_name]

                # Also store inference time
                if "avg_inference_time" in task_result:
                    metrics[f"system_{system_type}_{task}_time"] = task_result["avg_inference_time"]

    return metrics

def generate_comparison_tables(metrics):
    """Generate comparison tables for each task"""
    tables = {}

    # Define tasks and their metrics
    tasks = ["summarization", "qa", "paraphrase"]
    task_metrics = {
        "summarization": ["rouge_l"],
        "qa": ["rouge_l", "bertscore"],
        "paraphrase": ["sacrebleu", "meteor"]
    }

    # Models and systems to compare
    models = ["qwen", "opt", "llama"]
    systems = ["dynamic", "ensemble", "pipeline"]

    for task in tasks:
        table_data = []
        headers = ["System/Model"] + [m.upper() for m in task_metrics[task]] + ["Time (s)"]

        # Add individual models
        for model in models:
            row = [model.upper()]
            for metric in task_metrics[task]:
                key = f"{model}_{task}_{metric}"
                value = metrics.get(key, "N/A")
                row.append(value if value != "N/A" else "N/A")

            # Add time
            time_key = f"{model}_{task}_time"
            time_value = metrics.get(time_key, "N/A")
            row.append(time_value if time_value != "N/A" else "N/A")

            table_data.append(row)

        # Add multi-model systems
        for system in systems:
            row = [f"{system.upper()} SYSTEM"]
            for metric in task_metrics[task]:
                key = f"system_{system}_{task}_{metric}"
                value = metrics.get(key, "N/A")
                row.append(value if value != "N/A" else "N/A")

            # Add time
            time_key = f"system_{system}_{task}_time"
            time_value = metrics.get(time_key, "N/A")
            row.append(time_value if time_value != "N/A" else "N/A")

            table_data.append(row)

        tables[task] = tabulate(table_data, headers=headers, tablefmt="grid")

    return tables

def generate_plots(metrics, output_dir):
    """Generate comparison plots for visualization"""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Define tasks and their metrics
    tasks = ["summarization", "qa", "paraphrase"]
    task_metrics = {
        "summarization": ["rouge_l"],
        "qa": ["rouge_l", "bertscore"],
        "paraphrase": ["sacrebleu", "meteor"]
    }

    # Models and systems to compare
    models = ["qwen", "opt", "llama"]
    systems = ["dynamic", "ensemble", "pipeline"]

    for task in tasks:
        for metric in task_metrics[task]:
            plt.figure(figsize=(10, 6))

            # Collect data for models
            model_names = []
            model_values = []
            for model in models:
                key = f"{model}_{task}_{metric}"
                if key in metrics:
                    model_names.append(model.upper())
                    model_values.append(metrics[key])

            # Collect data for systems
            system_names = []
            system_values = []
            for system in systems:
                key = f"system_{system}_{task}_{metric}"
                if key in metrics:
                    system_names.append(f"{system.upper()} SYSTEM")
                    system_values.append(metrics[key])

            # Plot bars
            x = np.arange(len(model_names) + len(system_names))
            plt.bar(x[:len(model_names)], model_values, color='skyblue', label='Individual Models')
            plt.bar(x[len(model_names):], system_values, color='orange', label='Multi-Model Systems')

            # Add labels and title
            plt.xlabel('Model / System')
            plt.ylabel(f'{metric.upper()} Score')
            plt.title(f'{metric.upper()} Comparison for {task.capitalize()}')
            plt.xticks(x, model_names + system_names, rotation=45, ha='right')
            plt.legend()
            plt.tight_layout()

            # Save plot
            plot_path = os.path.join(output_dir, f"{task}_{metric}_comparison.png")
            plt.savefig(plot_path)
            logger.info(f"Saved plot to {plot_path}")
            plt.close()

def main(args):
    # Load results
    results = load_results(args.results_dir)

    # Calculate metrics
    metrics = calculate_metrics(results)

    # Generate comparison tables
    tables = generate_comparison_tables(metrics)

    # Print tables
    for task, table in tables.items():
        logger.info(f"\nComparison for {task.upper()}:")
        logger.info(f"\n{table}")

    # Generate plots
    generate_plots(metrics, args.results_dir)

    # Save final comparison
    output = {
        "metrics": metrics,
        "tables": {task: table.split('\n') for task, table in tables.items()}
    }

    with open(args.output_file, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"Saved final comparison to {args.output_file}")

if __name__ == "__main__":
    args = parse_args()
    main(args)
