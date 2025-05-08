import torch
import time
# import os
import random
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluate import load as load_metric
import nltk # Required for METEOR

# --- Configuration ---

# Model identifiers
model_identifiers = {
    "Qwen": "Qwen/Qwen2.5-1.5B",
    "OPT": "facebook/opt-1.3b",
    "Llama": "meta-llama/Llama-3.2-1B"
}


# Model loading arguments (use the same successful config as before)
# Example using 4-bit quantization
model_load_args = {
    "device_map": "auto",
#     "load_in_4bit": True,
#     "bnb_4bit_quant_type": "nf4",
#     "bnb_4bit_compute_dtype": torch.bfloat16,
    "torch_dtype": torch.bfloat16,
    "trust_remote_code": True # Keep if needed for Qwen/Llama
}
# Ensure accelerate and bitsandbytes are installed

# --- Data Loading and Sampling ---
NUM_SAMPLES = 300 # Number of samples per dataset for baseline (adjust as needed)
CACHE_DIR = "~/.cache" # Optional: specify cache

print("Loading and sampling datasets...")
datasets_subset = {}

# Summarization (CNN/DailyMail)
cnn_dm = load_dataset("cnn_dailymail", '3.0.0', split='train', cache_dir=CACHE_DIR)
cnn_dm_subset = cnn_dm.shuffle(seed=42).select(range(NUM_SAMPLES))
datasets_subset['summarization'] = cnn_dm_subset
print(f"Loaded {len(datasets_subset['summarization'])} samples from CNN/DailyMail.")

# Question Answering (SQuAD v2)
squad = load_dataset("squad_v2", split='train', cache_dir=CACHE_DIR)
# Filter out examples with no possible answers for simpler baseline generation? Optional.
# squad_filtered = squad.filter(lambda example: len(example['answers']['text']) > 0)
squad_subset = squad.shuffle(seed=42).select(range(NUM_SAMPLES))
datasets_subset['qa'] = squad_subset
print(f"Loaded {len(datasets_subset['qa'])} samples from SQuAD v2.")

# Paraphrase Generation (Quora) - Needs preprocessing
# Quora dataset structure is pairs of questions. We need to select one as input, one as reference.
quora = load_dataset("quora", split='train', cache_dir=CACHE_DIR)
# Filter out potential duplicates if necessary, or just sample
quora_processed = []
quora_indices = random.sample(range(len(quora)), k=NUM_SAMPLES * 2) # Sample more initially
added_count = 0
seen_pairs = set()
for i in quora_indices:
    pair = quora[i]['questions']
    # Ensure pair ID is unique regardless of order to avoid duplicates
    pair_id = tuple(sorted((pair['id'][0], pair['id'][1])))
    if pair['id'][0] is not None and pair['id'][1] is not None and pair_id not in seen_pairs:
         # Treat question 0 as input, question 1 as reference paraphrase
        quora_processed.append({
            'input_question': pair['text'][0],
            'reference_paraphrase': pair['text'][1]
        })
        seen_pairs.add(pair_id)
        added_count += 1
        if added_count >= NUM_SAMPLES:
            break

# Alternative: Treat question 1 as input, question 0 as reference
# for i in quora_indices:
#      pair = quora[i]['questions']
#      pair_id = tuple(sorted((pair['id'][0], pair['id'][1])))
#      if pair['id'][0] is not None and pair['id'][1] is not None and pair_id not in seen_pairs:
#          quora_processed.append({
#              'input_question': pair['text'][1],
#              'reference_paraphrase': pair['text'][0]
#          })
#          seen_pairs.add(pair_id)
#          added_count += 1
#          if added_count >= NUM_SAMPLES:
#              break

datasets_subset['paraphrase'] = quora_processed # Store as a list of dicts
print(f"Processed {len(datasets_subset['paraphrase'])} samples from Quora.")

# --- Define Prompts ---
def get_prompt(task_name, example):
    if task_name == "summarization":
        # Simple instruction prompt
        # Note: Truncate article if too long for model context
        max_article_len = 1024 # Adjust based on model context length & typical article size
        article = example['article'][:max_article_len]
        return f"Summarize the following news article:\n\n{article}\n\nSummary:"
    elif task_name == "qa":
        # SQuAD prompt format
        context = example['context']
        question = example['question']
        # Truncate context if needed
        max_context_len = 1024 # Adjust
        context = context[:max_context_len]
        # Add instruction for unanswerable questions
        return f"Based on the context below, answer the question. If the context does not provide the answer, respond with 'unanswerable'.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
    elif task_name == "paraphrase":
        sentence = example['input_question']
        return f"Generate a paraphrase for the following sentence:\n\nSentence: {sentence}\n\nParaphrase:"
    else:
        raise ValueError(f"Unknown task: {task_name}")

# --- Evaluation Metrics ---
print("Loading evaluation metrics...")
rouge = load_metric("rouge")
bertscore = load_metric("bertscore")
sacrebleu = load_metric("sacrebleu")
meteor = load_metric("meteor")

# Download WordNet for METEOR if not already done
try:
    nltk.data.find('corpora/wordnet')
    nltk.data.find('tokenizers/punkt') # Check for punkt as well
    nltk.data.find('corpora/omw-1.4') # Check for omw-1.4 as well
    print("NLTK data (wordnet, punkt, omw-1.4) found.")
except LookupError: # <-- Catch the actual error
    print("Downloading NLTK data (wordnet, punkt, omw-1.4)...")
    try:
        nltk.download('wordnet', quiet=True) # Add quiet=True to reduce verbose output if desired
        nltk.download('punkt', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        print("NLTK data downloaded successfully.")
        # Optional: Verify again after download
        nltk.data.find('corpora/wordnet')
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/omw-1.4')
    except Exception as download_err: # Catch potential errors during download itself
        print(f"ERROR: Failed to download NLTK data: {download_err}")
        print("Please try downloading manually:")
        print("1. Activate environment: conda activate nlp_new")
        print("2. Run python")
        print("3. Inside python, run: import nltk; nltk.download('wordnet'); nltk.download('punkt'); nltk.download('omw-1.4')")
        # Exit or raise to stop the script if NLTK data is essential
        raise RuntimeError("NLTK data is required but download failed.") from download_err

# --- Inference and Evaluation Loop ---
results = {} # Store {model_name: {task_name: {metrics: ..., time: ...}}}

# Generation parameters (adjust as needed per task/model)
# These are crucial for quality! Experimentation needed later.
generation_config = {
    "max_new_tokens": 150,  # Default, override per task
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.9,
    # "num_beams": 4, # Beam search can be slower but potentially better
    # "early_stopping": True,
}

task_specific_max_tokens = {
    "summarization": 256, # Allow longer summaries
    "qa": 50,            # Answers are usually short
    "paraphrase": 100
}


for model_name, model_id in model_identifiers.items():
    print(f"\n===== Processing Model: {model_name} ({model_id}) =====")
    results[model_name] = {}

    try:
        # Load Model and Tokenizer (ONE AT A TIME)
        print(f"Loading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        # Add padding token if missing (common issue for some models like Llama)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"   - Added EOS token as PAD token.")

        model = AutoModelForCausalLM.from_pretrained(model_id, **model_load_args)
        model.config.pad_token_id = tokenizer.pad_token_id # Ensure model config matches

        print(f"   - Model and Tokenizer loaded.")
        model_load_success = True
    except Exception as e:
        print(f"SKIPPING MODEL {model_name} due to loading error: {e}")
        model_load_success = False
        results[model_name] = "Loading Failed"
        continue # Skip to the next model

    if model_load_success:
        for task_name, dataset in datasets_subset.items():
            print(f"\n--- Task: {task_name} ---")
            results[model_name][task_name] = {}
            predictions = []
            references = []
            total_inference_time = 0

            # Prepare references based on task
            if task_name == "summarization":
                references = [ex['highlights'] for ex in dataset]
            elif task_name == "qa":
                # SQuAD references are lists of possible answers or empty if unanswerable
                references = [ex['answers']['text'] if len(ex['answers']['text']) > 0 else ["unanswerable"] for ex in dataset]
            elif task_name == "paraphrase":
                references = [ex['reference_paraphrase'] for ex in dataset]

            # Update generation config for the task
            current_gen_config = generation_config.copy()
            current_gen_config["max_new_tokens"] = task_specific_max_tokens.get(task_name, generation_config["max_new_tokens"])

            print(f"Generating {len(dataset)} predictions...")
            start_task_time = time.time()
            for i, example in enumerate(dataset):
                prompt = get_prompt(task_name, example)
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device) # Move inputs to GPU

                # Handle potential empty inputs after truncation (edge case)
                if inputs.input_ids.shape[1] == 0:
                    print(f"Warning: Empty input after tokenization for example {i}. Skipping.")
                    predictions.append("") # Or handle as appropriate
                    continue

                start_inference = time.time()
                with torch.no_grad(): # Important for inference
                    outputs = model.generate(
                        **inputs,
                        **current_gen_config,
                        pad_token_id=tokenizer.pad_token_id # Explicitly set pad token id
                    )
                end_inference = time.time()
                total_inference_time += (end_inference - start_inference)

                # Decode only the newly generated tokens
                input_length = inputs.input_ids.shape[1]
                generated_ids = outputs[0, input_length:]
                prediction = tokenizer.decode(generated_ids, skip_special_tokens=True)

                # Basic cleanup (can be improved)
                prediction = prediction.strip()
                predictions.append(prediction)

                if (i + 1) % 20 == 0: # Print progress
                   print(f"   Processed {i+1}/{len(dataset)} samples...")

            end_task_time = time.time()
            avg_inference_time = total_inference_time / len(dataset) if len(dataset) > 0 else 0
            print(f"Generation finished. Average inference time: {avg_inference_time:.4f} sec/sample.")

            # Calculate Metrics
            print("Calculating metrics...")
            task_metrics = {}
            task_metrics['avg_inference_time_sec'] = avg_inference_time

            try:
                if task_name == "summarization":
                    rouge_results = rouge.compute(predictions=predictions, references=references)
                    task_metrics['ROUGE-L'] = rouge_results['rougeL']
#                 elif task_name == "qa":
#                     # ROUGE-L for QA (treat reference list as single possibilities for simple ROUGE)
#                     # A better approach might involve comparing against *each* reference if > 1
#                     simple_references_for_rouge = [" ".join(ref_list) for ref_list in references]
#                     rouge_results = rouge.compute(predictions=predictions, references=simple_references_for_rouge)
#                     task_metrics['ROUGE-L'] = rouge_results['rougeL']

#                     # BERTScore for QA (can handle multiple references better)
#                     # Note: BERTScore can be slow, might need GPU. Check device placement.
#                     # BERTScore expects lists of strings for predictions and references.
#                     # For QA, references is a list of lists. We might need to adjust how it's passed
#                     # depending on the specific implementation detail of evaluate's bertscore.
#                     # A common way is to compute score against each ref and take max, but evaluate might handle list-of-lists.
#                     # Let's try passing it directly first.
#                     # We need actual string references for bertscore, not the potentially 'unanswerable' placeholder.
#                     bertscore_references = [ex['answers']['text'] for ex in dataset] # Get original answer lists
#                     # Check if bertscore compute can handle list of lists for references. If not, manual loop needed.
#                     try:
#                        bertscore_results = bertscore.compute(predictions=predictions, references=bertscore_references, lang="en", device=model.device)
#                        # Average the F1 score
#                        task_metrics['BERTScore_F1'] = sum(bertscore_results['f1']) / len(bertscore_results['f1'])
#                     except Exception as b_err:
#                        print(f"  - BERTScore calculation failed: {b_err}. Check reference format / device.")
#                        task_metrics['BERTScore_F1'] = "Error"
                
                elif task_name == "qa":
                    # ROUGE-L calculation (remains the same)
                    simple_references_for_rouge = [" ".join(ref_list) if ref_list else "unanswerable" for ref_list in references]
                    rouge_results = rouge.compute(predictions=predictions, references=simple_references_for_rouge)
                    task_metrics['ROUGE-L'] = rouge_results['rougeL']

                    # BERTScore for QA - ** MODIFIED SECTION **
                    print("Preparing data for BERTScore...")
                    bertscore_references_original = [ex['answers']['text'] for ex in dataset] # Original list of lists

                    filtered_predictions = []
                    filtered_references = []
                    valid_indices = [] # Keep track of which samples had answers

                    for i, (pred, ref_list) in enumerate(zip(predictions, bertscore_references_original)):
                        # Keep only pairs where the reference list is NOT empty
                        if ref_list: # Checks if the list is non-empty
                            filtered_predictions.append(pred)
                            filtered_references.append(ref_list) # Keep as list of lists for bertscore
                            valid_indices.append(i)

                    print(f"Calculating BERTScore for {len(filtered_predictions)} samples with answers...")
                    if filtered_predictions: # Only compute if there are answerable questions in the sample
                        try:
                           # Ensure bertscore uses the correct device (e.g., GPU if available)
                           bertscore_device = model.device if hasattr(model, 'device') else 'cuda' if torch.cuda.is_available() else 'cpu'
                           print(f"Using device {bertscore_device} for BERTScore calculation.")

                           bertscore_results = bertscore.compute(
                               predictions=filtered_predictions,
                               references=filtered_references, # Pass the filtered list of lists
                               lang="en", # Use default roberta-large model
                               device=bertscore_device,
                               # Optional: Increase batch size if memory allows for faster computation
                               # batch_size=16
                           )
                           # Average the F1 score
                           task_metrics['BERTScore_F1'] = sum(bertscore_results['f1']) / len(bertscore_results['f1'])
                           print("BERTScore calculated successfully.")
                        except Exception as b_err:
                           print(f"  - BERTScore calculation failed unexpectedly: {b_err}")
                           task_metrics['BERTScore_F1'] = "Error"
                    else:
                        print("  - Skipping BERTScore: No examples with answers found in this subset.")
                        task_metrics['BERTScore_F1'] = "N/A (No answerable samples)"
                    # ** END MODIFIED SECTION **

                elif task_name == "paraphrase":
                    # SacreBLEU expects references to be a list of lists of strings
                    sacrebleu_references = [[ref] for ref in references] # Wrap each ref in a list
                    sacrebleu_results = sacrebleu.compute(predictions=predictions, references=sacrebleu_references)
                    task_metrics['SacreBLEU'] = sacrebleu_results['score']

                    # METEOR
                    meteor_results = meteor.compute(predictions=predictions, references=references)
                    task_metrics['METEOR'] = meteor_results['meteor']

                results[model_name][task_name] = task_metrics
                print(f"Metrics for {model_name} on {task_name}: {task_metrics}")

            except Exception as eval_err:
                print(f"ERROR calculating metrics for {task_name}: {eval_err}")
                results[model_name][task_name] = {"Error": str(eval_err)}


        # --- IMPORTANT: Clean up GPU memory before loading next model ---
        print(f"\nUnloading {model_name} to free memory...")
        del model
        del tokenizer
        torch.cuda.empty_cache()
        print("   - Memory cleared.")
        # -------------------------------------------------------------

# --- Display Results ---
print("\n\n===== Baseline Performance Summary =====")
for model_name, tasks in results.items():
    print(f"\n--- Model: {model_name} ---")
    if isinstance(tasks, str): # Handle loading failures
        print(f"   {tasks}")
        continue
    for task_name, metrics in tasks.items():
        print(f"  Task: {task_name}")
        if isinstance(metrics, dict):
            for metric_name, score in metrics.items():
                 if isinstance(score, float):
                     print(f"    {metric_name}: {score:.4f}")
                 else:
                     print(f"    {metric_name}: {score}")
        else:
             print(f"    Metrics calculation failed.")

print("\nBaseline evaluation complete.")