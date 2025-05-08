from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# MODEL_DIR = os.path.expanduser("~/NLP_Project/models")  # resolves ~ to full path
# os.makedirs(MODEL_DIR, exist_ok=True)

models_to_download = {
    "facebook/opt-1.3b": {
        "tokenizer_kwargs": {"use_fast": False},
        "model_cls": AutoModelForCausalLM,
        "tokenizer_cls": AutoTokenizer,
    },
    "meta-llama/Llama-3.2-1B": {
        "tokenizer_kwargs": {},
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

for model_name, config in models_to_download.items():
    print(f"Downloading {model_name}...")
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
        print(f"Successfully downloaded {model_name}")
    except Exception as e:
        print(f"Error downloading {model_name}: {e}")
    print("-" * 20)
