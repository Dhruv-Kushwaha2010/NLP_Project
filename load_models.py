from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import logging
from device_utils import get_device, prepare_model_kwargs

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_loading.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set up model directory
MODEL_DIR = os.path.expanduser("~/NLP_Project/models")  # resolves ~ to full path
os.makedirs(MODEL_DIR, exist_ok=True)

models_to_download = {
    "facebook/opt-1.3b": {
        "tokenizer_kwargs": {"use_fast": False},
        "model_kwargs": {},
        "model_cls": AutoModelForCausalLM,
        "tokenizer_cls": AutoTokenizer,
    },
    "meta-llama/Llama-3.2-1B": {
        "tokenizer_kwargs": {"token": "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"},
        "model_kwargs": {"token": "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"},
        "model_cls": "LlamaForCausalLM",
        "tokenizer_cls": "LlamaTokenizer"
    },
    "Qwen/Qwen2.5-1.5B": {
        "tokenizer_kwargs": {"trust_remote_code": True},
        "model_kwargs": {"trust_remote_code": True},
        "model_cls": AutoModelForCausalLM,
        "tokenizer_cls": AutoTokenizer,
    }
}

def load_models(use_quantization=False):
    """
    Load all models with appropriate device settings

    Args:
        use_quantization (bool, optional): Whether to use 8-bit quantization. Defaults to False.

    Returns:
        dict: Dictionary of loaded models and tokenizers
    """
    device = get_device()
    logger.info(f"Loading models on {device}")

    loaded_models = {}

    for model_name, config in models_to_download.items():
        logger.info(f"Loading {model_name}...")
        try:
            if isinstance(config["tokenizer_cls"], str):
                from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM
                tokenizer_cls = eval(config["tokenizer_cls"])
                model_cls = eval(config["model_cls"])
            else:
                tokenizer_cls = config["tokenizer_cls"]
                model_cls = config["model_cls"]

            # Load tokenizer
            tokenizer = tokenizer_cls.from_pretrained(
                model_name,
                cache_dir=MODEL_DIR,
                **config["tokenizer_kwargs"]
            )

            # Ensure tokenizer has pad token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                logger.info(f"Set pad_token to eos_token for {model_name}")

            # Prepare model kwargs with device settings
            model_kwargs = prepare_model_kwargs(
                config.get("model_kwargs", {}),
                use_quantization=use_quantization,
                device=device
            )

            # Load model
            model = model_cls.from_pretrained(
                model_name,
                cache_dir=MODEL_DIR,
                **model_kwargs
            )

            # Store loaded model and tokenizer
            short_name = model_name.split('/')[-1].lower()
            loaded_models[short_name] = {
                "model": model,
                "tokenizer": tokenizer
            }

            logger.info(f"Successfully loaded {model_name}")
        except Exception as e:
            logger.error(f"Error loading {model_name}: {e}")

        logger.info("-" * 40)

    return loaded_models

if __name__ == "__main__":
    # When run directly, download all models
    for model_name, config in models_to_download.items():
        logger.info(f"Downloading {model_name}...")
        try:
            if isinstance(config["tokenizer_cls"], str):
                from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM
                tokenizer_cls = eval(config["tokenizer_cls"])
                model_cls = eval(config["model_cls"])
            else:
                tokenizer_cls = config["tokenizer_cls"]
                model_cls = config["model_cls"]

            tokenizer = tokenizer_cls.from_pretrained(model_name, cache_dir=MODEL_DIR, **config["tokenizer_kwargs"])
            model = model_cls.from_pretrained(model_name, cache_dir=MODEL_DIR, **config.get("model_kwargs", {}))
            logger.info(f"Successfully downloaded {model_name}")
        except Exception as e:
            logger.error(f"Error downloading {model_name}: {e}")
        logger.info("-" * 40)
