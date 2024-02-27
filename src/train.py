# $env:DEBUG=3; $env:CACHE_DIR="D:models"; $env:DEVICE="cuda"; python .\src\train.py
import os, torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers.models.gpt2.modeling_gpt2 import CausalLMOutputWithCrossAttentions
from transformers import TrainingArguments, Trainer, GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
from datasets import load_dataset, load_metric

from utils import CACHE_DIR, DEBUG, DEVICE, Timing
from models import GPT2Wrapper
from data import TextToSQLDataset

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


if __name__ == "__main__":
    if DEBUG >= 2: print("DEVICE:", DEVICE)
    with Timing("Loading model ", enabled=DEBUG >= 2):
        model = GPT2Wrapper().to(DEVICE)
    if DEBUG >= 2: print(model)

    model = GPT2LMHeadModel.from_pretrained("gpt2", cache_dir=CACHE_DIR).to(DEVICE)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2", cache_dir=CACHE_DIR)

    metric = load_metric("accuracy")

    dataset = load_dataset("kaxap/llama2-sql-instruct")
    # Truncate the dataset to 1000
    dataset = dataset["train"].select(range(1000))
    print(dataset)
    #tokenizer = model.tokenizer

    def tokenize_function(examples):
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer(examples["text"], padding="max_length", truncation=True, return_tensors="pt")

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    print(tokenized_datasets)

    small_train_dataset = tokenized_datasets.shuffle(seed=42).select(range(1000))
    small_eval_dataset = tokenized_datasets.shuffle(seed=42).select(range(100))

    training_args = TrainingArguments(
        output_dir="test_trainer",
        evaluation_strategy="epoch",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
