import os
import datasets
from datasets import load_dataset
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
cache_dir = "data/gsm8k"

dataset = load_dataset("openai/gsm8k", "main", cache_dir=cache_dir)
print(dataset['train'][0])

def format_data(example):
    return {
        "input": example["question"],
        "output": example["answer"]
    }

formatted_data = dataset.map(format_data)
formatted_data.to_json("data/gsm8k_sft.json")