from datasets import load_dataset

# Define cache directory if needed (useful on HPC)
cache_dir = "~/.cache"

# Summarization
cnn_dailymail = load_dataset("cnn_dailymail", '3.0.0', split='train', cache_dir=cache_dir)

# Question Answering
squad = load_dataset("squad_v2", split='train', cache_dir=cache_dir)

# Paraphrase Generation
quora = load_dataset("quora", split='train', cache_dir=cache_dir)

print("Datasets loaded:")
print(cnn_dailymail)
print(squad)
print(quora)