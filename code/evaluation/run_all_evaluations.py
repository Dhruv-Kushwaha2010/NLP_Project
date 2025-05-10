#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to run all evaluations for all models and tasks
"""

import os
import argparse
import logging
import subprocess
import sys
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("all_evaluations.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define model-task combinations
MODEL_TASK_COMBINATIONS = [
    {"model": "qwen", "task": "summarization"},
    {"model": "opt", "task": "qa"},
    {"model": "llama", "task": "paraphrase"}
]

def parse_args():
    parser = argparse.ArgumentParser(description="Run all evaluations for all models and tasks")
    parser.add_argument("--models_dir", type=str, default="./fine_tuned_models",
                        help="Directory containing fine-tuned models")
    parser.add_argument("--num_samples", type=int, default=20,
                        help="Number of samples to evaluate")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results",
                        help="Directory to save evaluation results")
    parser.add_argument("--wait_time", type=int, default=30,
                        help="Wait time in seconds between evaluations")
    parser.add_argument("--combinations", type=str, nargs="+", default=None,
                        help="Specific model-task combinations to evaluate (format: 'model:task')")

    return parser.parse_args()

def run_evaluation(model, task, args):
    """Run evaluation for a specific model and task"""
    logger.info(f"Evaluating {model} on {task}")

    # Create output file path
    output_file = os.path.join(args.output_dir, f"{model}_{task}_results.json")

    # Build command
    cmd = [
        "python", "evaluate_model.py",  # Changed from evaluate_model_wrapper.py to direct call
        "--model", model,
        "--task", task,
        "--models_dir", args.models_dir,
        "--num_samples", str(args.num_samples),
        "--output_file", output_file,
        "--seed", "42"  # Added fixed seed for reproducibility
    ]

    # Log the command
    logger.info(f"Running command: {' '.join(cmd)}")

    # Run the command
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1  # Line buffered
        )

        # Stream output in real-time with progress indicator
        for line in process.stdout:
            print(line, end='')
            logger.info(line.strip())

        process.wait()

        if process.returncode == 0:
            logger.info(f"Evaluation completed successfully for {model} on {task}")
            return True
        else:
            logger.error(f"Evaluation failed with return code {process.returncode}")
            return False

    except Exception as e:
        logger.error(f"Error running evaluation: {e}")
        return False

def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Filter combinations if specified
    combinations_to_evaluate = MODEL_TASK_COMBINATIONS
    if args.combinations:
        filtered_combinations = []
        for combo_str in args.combinations:
            try:
                model, task = combo_str.split(":")
                filtered_combinations.append({"model": model, "task": task})
            except ValueError:
                logger.error(f"Invalid combination format: {combo_str}. Use 'model:task'")

        if filtered_combinations:
            combinations_to_evaluate = filtered_combinations

    # Track successful and failed evaluations
    successful_evals = []
    failed_evals = []

    # Run evaluation for each model and task
    for combo in combinations_to_evaluate:
        model = combo["model"]
        task = combo["task"]

        logger.info(f"=== Evaluating {model} on {task} ===")

        # Run evaluation
        success = run_evaluation(model, task, args)

        # Track result
        if success:
            successful_evals.append(f"{model}_{task}")
        else:
            failed_evals.append(f"{model}_{task}")

        # Wait between evaluations to allow memory cleanup
        if combo != combinations_to_evaluate[-1]:
            logger.info(f"Waiting {args.wait_time} seconds before next evaluation...")
            time.sleep(args.wait_time)

    # Print summary
    logger.info("\n=== Evaluation Summary ===")
    logger.info(f"Successful evaluations ({len(successful_evals)}): {', '.join(successful_evals)}")
    logger.info(f"Failed evaluations ({len(failed_evals)}): {', '.join(failed_evals)}")

    # Generate report summary
    logger.info("Generating report summary...")
    cmd = ["python", "generate_report_summary.py", "--results_dir", args.output_dir]
    try:
        subprocess.run(cmd, check=True)
        logger.info("Report summary generated successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error generating report summary: {e}")

    logger.info("All evaluations complete!")

    if failed_evals:
        return 1
    else:
        return 0

if __name__ == "__main__":
    sys.exit(main())
