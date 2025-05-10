import os
import torch
import argparse
import logging
import math
import sys
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed
)
from tqdm.auto import tqdm
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
import wandb

# Add environment variable to disable bitsandbytes welcome message
os.environ["BITSANDBYTES_NOWELCOME"] = "1"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("fine_tuning.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Fix for bitsandbytes on HPC systems
def setup_cuda_libraries():
    """Setup CUDA libraries for bitsandbytes on HPC systems."""
    # Common CUDA library paths on HPC systems
    cuda_paths = [
        "/usr/local/cuda/lib64",
        "/usr/local/cuda-12.4/targets/x86_64-linux/lib",  # Updated for CUDA 12.4
        "/usr/local/cuda-12/lib64",
        "/usr/local/cuda-11.8/targets/x86_64-linux/lib",  # Updated for CUDA 11.8
        "/usr/local/cuda-11/lib64",
        "/usr/local/cuda-12.0/lib64",
        "/usr/local/cuda-11.7/lib64",
        "/usr/local/cuda-11.6/lib64",
        "/opt/cuda/lib64",
    ]

    # Add paths to LD_LIBRARY_PATH if they exist
    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    new_paths = []

    for path in cuda_paths:
        if os.path.exists(path) and path not in ld_library_path:
            new_paths.append(path)

    if new_paths:
        os.environ["LD_LIBRARY_PATH"] = ld_library_path + ":" + ":".join(new_paths)
        logger.info(f"Added CUDA paths to LD_LIBRARY_PATH: {new_paths}")
        # We won't restart the process, as it can cause issues
        # Instead, we'll set environment variables to help bitsandbytes find the libraries
        os.environ["BNB_CUDA_VERSION"] = ""  # Clear any existing value
        logger.info("Set environment variables for bitsandbytes")

# Try to import bitsandbytes, but don't fail if it's not available
try:
    import bitsandbytes as bnb
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False
    logger.warning("bitsandbytes not available, quantization will be disabled")
except RuntimeError as e:
    if "CUDA Setup failed" in str(e):
        logger.warning("CUDA setup for bitsandbytes failed. Attempting to fix...")
        setup_cuda_libraries()
    BITSANDBYTES_AVAILABLE = False
    logger.warning("bitsandbytes CUDA setup failed, quantization will be disabled")

# Define model configurations
MODEL_CONFIGS = {
    "qwen": {
        "model_id": "Qwen/Qwen2.5-1.5B",
        "tokenizer_kwargs": {"trust_remote_code": True},
        "model_kwargs": {"trust_remote_code": True},
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
    },
    "opt": {
        "model_id": "facebook/opt-1.3b",
        "tokenizer_kwargs": {"use_fast": False},
        "model_kwargs": {},
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
    },
    "llama": {
        "model_id": "meta-llama/Llama-3.2-1B",
        "tokenizer_kwargs": {"token": "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"},
        "model_kwargs": {"token": "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"},
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
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
        "max_target_length": 256
    },
    "qa": {
        "dataset": "squad_v2",
        "dataset_version": None,
        "prompt_template": "Based on the context below, answer the question. If the context does not provide the answer, respond with 'unanswerable'.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:",
        "input_columns": ["context", "question"],
        "target_column": "answers",
        "max_input_length": 1024,
        "max_target_length": 128
    },
    "paraphrase": {
        "dataset": "quora",
        "dataset_version": None,
        "prompt_template": "Generate a paraphrase for the following sentence:\n\nSentence: {input_question}\n\nParaphrase:",
        "input_column": "input_question",
        "target_column": "reference_paraphrase",
        "max_input_length": 512,
        "max_target_length": 128
    }
}

# Dictionary to store dataset sizes
DATASET_SIZES = {
    "summarization": 287113,  # CNN/DailyMail
    "qa": 130319,            # SQuAD v2
    "paraphrase": 404290     # Quora
}

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune language models with PEFT (LoRA)")
    parser.add_argument("--model", type=str, required=True, choices=["qwen", "opt", "llama"],
                        help="Model to fine-tune")
    parser.add_argument("--task", type=str, required=True, choices=["summarization", "qa", "paraphrase"],
                        help="Task to fine-tune on")
    parser.add_argument("--output_dir", type=str, default="./fine_tuned_models",
                        help="Directory to save the fine-tuned model")
    parser.add_argument("--num_train_samples", type=int, default=100,
                        help="Number of training samples to use")
    parser.add_argument("--num_eval_samples", type=int, default=20,
                        help="Number of evaluation samples to use")
    parser.add_argument("--dataset_size_percentage", type=float, default=None,
                        help="Percentage of dataset to use (0-100). Overrides num_train_samples if provided.")
    parser.add_argument("--lora_r", type=int, default=8,
                        help="LoRA attention dimension")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout probability")
    parser.add_argument("--learning_rate", type=float, default=5e-4,
                        help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Number of gradient accumulation steps")
    parser.add_argument("--num_epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--use_4bit", action="store_true",
                        help="Use 4-bit quantization")
    parser.add_argument("--use_8bit", action="store_true",
                        help="Use 8-bit quantization")
    parser.add_argument("--use_wandb", action="store_true",
                        help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="nlp-multi-model",
                        help="Weights & Biases project name")
    parser.add_argument("--wandb_name", type=str, default=None,
                        help="Weights & Biases run name")

    args = parser.parse_args()

    # Print dataset sizes when --help is called
    if hasattr(args, 'help') and args.help:
        print("\nDataset Sizes:")
        print(f"  Summarization (CNN/DailyMail): {DATASET_SIZES['summarization']} samples")
        print(f"  Question Answering (SQuAD v2): {DATASET_SIZES['qa']} samples")
        print(f"  Paraphrase Generation (Quora): {DATASET_SIZES['paraphrase']} samples")

    return args

def prepare_dataset_for_summarization(dataset, tokenizer, config, num_train, num_eval):
    def preprocess_function(examples):
        inputs = [config["prompt_template"].format(article=article) for article in examples[config["input_column"]]]

        # Use the same max_length for both inputs and labels to avoid size mismatch
        max_length = min(config["max_input_length"], 512)  # Limit to 512 for memory constraints

        # Process inputs
        model_inputs = tokenizer(inputs, max_length=max_length, truncation=True, padding="max_length")

        # Process targets (labels)
        labels = tokenizer(examples[config["target_column"]], max_length=max_length,
                          truncation=True, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # Split dataset
    logger.info(f"Shuffling and splitting dataset for summarization")
    dataset = dataset.shuffle(seed=args.seed)
    train_dataset = dataset.select(range(num_train))
    eval_dataset = dataset.select(range(num_train, num_train + num_eval))

    # Preprocess datasets with progress bar
    logger.info(f"Preprocessing {len(train_dataset)} training examples")
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        desc="Preprocessing training dataset"
    )

    logger.info(f"Preprocessing {len(eval_dataset)} evaluation examples")
    eval_dataset = eval_dataset.map(
        preprocess_function,
        batched=True,
        desc="Preprocessing evaluation dataset"
    )

    return train_dataset, eval_dataset

def prepare_dataset_for_qa(dataset, tokenizer, config, num_train, num_eval):
    def preprocess_function(examples):
        inputs = [
            config["prompt_template"].format(context=context, question=question)
            for context, question in zip(examples["context"], examples["question"])
        ]

        # Use the same max_length for both inputs and labels to avoid size mismatch
        max_length = min(config["max_input_length"], 512)  # Limit to 512 for memory constraints

        model_inputs = tokenizer(inputs, max_length=max_length, truncation=True, padding="max_length")

        # Process answers (handling unanswerable questions)
        targets = []
        for answer in examples["answers"]:
            if answer["text"]:  # If there's at least one answer
                targets.append(answer["text"][0])  # Take the first answer
            else:
                targets.append("unanswerable")

        # Tokenize targets
        labels = tokenizer(targets, max_length=max_length, truncation=True, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # Split dataset
    logger.info(f"Shuffling and splitting dataset for QA")
    dataset = dataset.shuffle(seed=args.seed)
    train_dataset = dataset.select(range(num_train))
    eval_dataset = dataset.select(range(num_train, num_train + num_eval))

    # Preprocess datasets with progress bar
    logger.info(f"Preprocessing {len(train_dataset)} training examples")
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        desc="Preprocessing QA training dataset"
    )

    logger.info(f"Preprocessing {len(eval_dataset)} evaluation examples")
    eval_dataset = eval_dataset.map(
        preprocess_function,
        batched=True,
        desc="Preprocessing QA evaluation dataset"
    )

    return train_dataset, eval_dataset

def prepare_dataset_for_paraphrase(dataset, tokenizer, config, num_train, num_eval):
    # Process Quora dataset to get input-reference pairs
    logger.info(f"Processing Quora dataset for paraphrase task")
    processed_data = []
    seen_pairs = set()

    # Use tqdm for progress tracking
    for i in tqdm(range(len(dataset)), desc="Processing Quora pairs"):
        pair = dataset[i]['questions']
        pair_id = tuple(sorted((pair['id'][0], pair['id'][1])))

        if pair['id'][0] is not None and pair['id'][1] is not None and pair_id not in seen_pairs:
            processed_data.append({
                'input_question': pair['text'][0],
                'reference_paraphrase': pair['text'][1]
            })
            seen_pairs.add(pair_id)

            if len(processed_data) >= (num_train + num_eval):
                break

    logger.info(f"Extracted {len(processed_data)} paraphrase pairs")

    # Convert to dataset format
    from datasets import Dataset
    processed_dataset = Dataset.from_dict({
        'input_question': [item['input_question'] for item in processed_data],
        'reference_paraphrase': [item['reference_paraphrase'] for item in processed_data]
    })

    def preprocess_function(examples):
        inputs = [config["prompt_template"].format(input_question=question)
                 for question in examples[config["input_column"]]]

        # Use the same max_length for both inputs and labels to avoid size mismatch
        max_length = min(config["max_input_length"], 512)  # Limit to 512 for memory constraints

        # Process inputs
        model_inputs = tokenizer(inputs, max_length=max_length, truncation=True, padding="max_length")

        # Process targets (labels)
        labels = tokenizer(examples[config["target_column"]], max_length=max_length,
                          truncation=True, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # Split dataset
    logger.info(f"Shuffling and splitting dataset for paraphrase")
    processed_dataset = processed_dataset.shuffle(seed=args.seed)
    train_dataset = processed_dataset.select(range(num_train))
    eval_dataset = processed_dataset.select(range(num_train, num_train + num_eval))

    # Preprocess datasets with progress bar
    logger.info(f"Preprocessing {len(train_dataset)} training examples")
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        desc="Preprocessing paraphrase training dataset"
    )

    logger.info(f"Preprocessing {len(eval_dataset)} evaluation examples")
    eval_dataset = eval_dataset.map(
        preprocess_function,
        batched=True,
        desc="Preprocessing paraphrase evaluation dataset"
    )

    return train_dataset, eval_dataset

def main(args):
    set_seed(args.seed)

    # Disable Sliding Window Attention for Qwen via environment variable
    if args.model == "qwen":
        os.environ["DISABLE_SLIDING_WINDOW_ATTENTION"] = "true"
        logger.info("Disabled Sliding Window Attention for Qwen via environment variable")

    # Setup CUDA libraries for HPC
    setup_cuda_libraries()

    # Create output directory
    model_task_dir = f"{args.output_dir}/{args.model}_{args.task}"
    os.makedirs(model_task_dir, exist_ok=True)

    # Initialize wandb if requested
    if args.use_wandb:
        wandb_run_name = args.wandb_name or f"{args.model}_{args.task}_lora"
        wandb.init(
            project=args.wandb_project,
            name=wandb_run_name,
            config={
                "model": args.model,
                "task": args.task,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "num_epochs": args.num_epochs,
                "seed": args.seed,
                "use_4bit": args.use_4bit,
                "use_8bit": args.use_8bit,
                "dataset_size_percentage": args.dataset_size_percentage
            }
        )
        logger.info(f"Initialized Weights & Biases logging: {wandb_run_name}")

    # Load model configuration
    model_config = MODEL_CONFIGS[args.model]
    task_config = TASK_CONFIGS[args.task]

    logger.info(f"Fine-tuning {args.model} on {args.task}")
    logger.info(f"Loading model: {model_config['model_id']}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["model_id"],
        **model_config["tokenizer_kwargs"]
    )

    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token")

    # Import device utilities
    from device_utils import get_device, prepare_model_kwargs, move_model_to_device

    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")

    # Load model with appropriate device settings
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

    # Load the model
    logger.info(f"Loading model: {model_config['model_id']}")
    model = AutoModelForCausalLM.from_pretrained(
        model_config["model_id"],
        **model_kwargs
    )

    # Move model to device (especially important for MPS)
    model = move_model_to_device(model, device)

    # Disable Sliding Window Attention for Qwen model
    if args.model == "qwen":
        logger.info("Disabling Sliding Window Attention for Qwen model")
        if hasattr(model.config, "sliding_window"):
            model.config.sliding_window = None
            logger.info("Disabled sliding_window in model config")
        # Also try to disable it in the model's attention modules
        for module in model.modules():
            if hasattr(module, "sliding_window"):
                module.sliding_window = None
                logger.info("Disabled sliding_window in attention module")

    # Prepare model for k-bit training if using quantization
    if (args.use_4bit or args.use_8bit) and BITSANDBYTES_AVAILABLE:
        try:
            model = prepare_model_for_kbit_training(model)
        except Exception as e:
            logger.warning(f"Failed to prepare model for k-bit training: {e}")
            logger.warning("Continuing without quantization")
    elif args.use_4bit or args.use_8bit:
        logger.warning("Bitsandbytes not available, continuing without quantization")

    # Configure LoRA
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=model_config["lora_target_modules"]
    )

    # Apply LoRA to model
    # Set environment variable to skip bitsandbytes import in PEFT
    os.environ["PEFT_SKIP_BNB_IMPORT"] = "1"

    try:
        model = get_peft_model(model, lora_config)
        try:
            model.print_trainable_parameters()
        except:
            # Print trainable parameters manually
            trainable_params = 0
            all_param = 0
            for _, param in model.named_parameters():
                all_param += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
            logger.info(
                f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
            )
    except Exception as e:
        logger.error(f"Error applying LoRA to model: {e}")
        logger.warning("Attempting alternative approach for LoRA...")

        # Try a different approach
        try:
            from peft.tuners.lora import LoraModel

            # Create a new LoraModel directly
            model = LoraModel(model, lora_config, "default")

            # Print trainable parameters manually
            trainable_params = 0
            all_param = 0
            for _, param in model.named_parameters():
                all_param += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
            logger.info(
                f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
            )
        except Exception as inner_e:
            logger.error(f"Alternative approach also failed: {inner_e}")
            logger.warning("Continuing with the original model without LoRA")

    # For MPS, ensure model is moved to device after LoRA application
    device = get_device()
    if device.type == "mps":
        logger.info("Moving LoRA model to MPS device")
        model = move_model_to_device(model, device)

    # Load and prepare dataset
    logger.info(f"Loading dataset: {task_config['dataset']}")

    # Handle specific dataset versions and configurations
    if task_config["dataset"] == "cnn_dailymail":
        logger.info("Loading CNN/DailyMail dataset with version 3.0.0")
        dataset = load_dataset("cnn_dailymail", "3.0.0", split="train")
    elif task_config["dataset"] == "squad_v2":
        logger.info("Loading SQuAD v2 dataset")
        dataset = load_dataset("squad_v2", split="train")
    elif task_config["dataset"] == "quora":
        logger.info("Loading Quora Question Pairs dataset")
        dataset = load_dataset("quora", split="train", trust_remote_code=True)
    else:
        # Fallback to generic loading
        dataset_args = {"split": "train"}
        if task_config["dataset_version"]:
            dataset_args["version"] = task_config["dataset_version"]
        dataset = load_dataset(task_config["dataset"], **dataset_args)

    # Calculate number of samples based on percentage if provided
    num_train_samples = args.num_train_samples
    if args.dataset_size_percentage is not None:
        total_dataset_size = DATASET_SIZES[args.task]
        num_train_samples = int(total_dataset_size * args.dataset_size_percentage / 100)
        logger.info(f"Using {args.dataset_size_percentage}% of dataset: {num_train_samples} training samples")

    logger.info(f"Total dataset size: {len(dataset)} samples")
    logger.info(f"Using {num_train_samples} training samples and {args.num_eval_samples} evaluation samples")

    # Prepare dataset based on task
    if args.task == "summarization":
        train_dataset, eval_dataset = prepare_dataset_for_summarization(
            dataset, tokenizer, task_config, num_train_samples, args.num_eval_samples
        )
    elif args.task == "qa":
        train_dataset, eval_dataset = prepare_dataset_for_qa(
            dataset, tokenizer, task_config, num_train_samples, args.num_eval_samples
        )
    elif args.task == "paraphrase":
        train_dataset, eval_dataset = prepare_dataset_for_paraphrase(
            dataset, tokenizer, task_config, num_train_samples, args.num_eval_samples
        )

    # Configure training arguments with wandb integration if requested
    report_to = ["wandb"] if args.use_wandb else "none"

    # Get device for memory optimization
    device = get_device()

    # Adjust settings based on device
    if device.type == "mps":
        # For MPS, use more conservative settings to avoid memory issues
        eval_steps = 50
        save_steps = 50
        logging_steps = 5
        fp16 = False  # MPS doesn't support mixed precision through Accelerator
        bf16 = False
        logger.info("Using MPS-optimized training settings")
    else:
        # Default settings for other devices
        eval_steps = 100
        save_steps = 100
        logging_steps = 10
        fp16 = False
        bf16 = device.type == "cuda"  # Use bfloat16 only on CUDA

    training_args = TrainingArguments(
        output_dir=model_task_dir,
        do_eval=True,
        eval_steps=eval_steps,
        save_steps=save_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_epochs,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_dir=f"{model_task_dir}/logs",
        logging_steps=logging_steps,
        save_total_limit=1,
        push_to_hub=False,
        report_to=report_to,
        fp16=fp16,
        bf16=bf16,
        # Memory optimizations
        optim="adamw_torch",  # Use PyTorch's AdamW which is more memory efficient
        dataloader_num_workers=1,  # Limit number of workers to reduce memory usage
        dataloader_pin_memory=False if device.type == "mps" else True,  # Disable pin_memory for MPS
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    # Train model
    logger.info("Starting training...")
    trainer.train()

    # Save model
    logger.info(f"Saving model to {model_task_dir}")
    model.save_pretrained(model_task_dir)
    tokenizer.save_pretrained(model_task_dir)

    # Finish wandb run if active
    if args.use_wandb and wandb.run is not None:
        wandb.finish()

    logger.info("Fine-tuning complete!")

if __name__ == "__main__":
    args = parse_args()
    main(args)
