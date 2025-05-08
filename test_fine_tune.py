import os
import torch
import logging
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType
)
from datasets import load_dataset
from device_utils import get_device, prepare_model_kwargs

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("test_fine_tune.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    # Set seed for reproducibility
    set_seed(42)

    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")

    # Create output directory
    output_dir = "./test_fine_tuned"
    os.makedirs(output_dir, exist_ok=True)

    # Load a smaller model - OPT-125M
    model_name = "facebook/opt-125m"
    logger.info(f"Loading model: {model_name}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token")

    # Prepare model kwargs with device settings
    model_kwargs = prepare_model_kwargs(
        {},
        use_quantization=False,
        device=device
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    # Configure LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load a small subset of CNN/DailyMail dataset
    logger.info("Loading CNN/DailyMail dataset")
    dataset = load_dataset("cnn_dailymail", "3.0.0", split="train[:10]")

    # Prepare dataset for summarization
    def preprocess_function(examples):
        prompt_template = "Summarize the following news article:\n\n{article}\n\nSummary:"
        inputs = [prompt_template.format(article=article) for article in examples["article"]]

        # Use the same max_length for both inputs and labels to avoid size mismatch
        max_length = 256

        # Process inputs
        model_inputs = tokenizer(inputs, max_length=max_length, truncation=True, padding="max_length")

        # Process targets (labels)
        labels = tokenizer(examples["highlights"], max_length=max_length, truncation=True, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    # Process dataset
    train_dataset = dataset.map(preprocess_function, batched=True)

    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=5e-4,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_dir=f"{output_dir}/logs",
        logging_steps=1,
        save_steps=5,
        save_total_limit=1,
        push_to_hub=False,
        report_to="none"
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    # Train model
    logger.info("Starting training...")
    trainer.train()

    # Save model
    logger.info(f"Saving model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info("Fine-tuning complete!")

if __name__ == "__main__":
    main()
