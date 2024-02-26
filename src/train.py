# $env:DEBUG=3; $env:CACHE_DIR="D:models"; $env:DEVICE="cuda"; python .\src\train.py
import os, torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers.models.gpt2.modeling_gpt2 import CausalLMOutputWithCrossAttentions

from utils import DEBUG, DEVICE, Timing
from models import GPT2Wrapper
from data import TextToSQLDataset


if __name__ == "__main__":
    if DEBUG >= 2: print("DEVICE:", DEVICE)
    with Timing("Loading model ", enabled=DEBUG >= 2):
        model = GPT2Wrapper().to(DEVICE)
    model.train()
    epochs = 10
    batch_size = 1
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    ds = os.path.join("F:", "tfg", "processed", "hf", "sql_context-train.csv")
    dataset = TextToSQLDataset(ds, model.tokenizer, model.max_length())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        t = tqdm(dataloader, desc="Batches")
        for tokens in t:
            print(type(tokens))
            tokens = tokens["input_ids"].to(DEVICE)
            print(tokens)
            outs: CausalLMOutputWithCrossAttentions = model(tokens)
            print(outs.logits)
            exit()
    print("Training finished")
