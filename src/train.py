# $env:DEBUG=3; $env:CACHE_DIR="D:models"; $env:DEVICE="cuda"; python .\src\train.py
import os
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from utils import DEBUG, DEVICE
from models import GPT2Wrapper
from data import TextToSQLDataset


if __name__ == "__main__":
    if DEBUG >= 2: print("DEVICE:", DEVICE)
    model = GPT2Wrapper().to(DEVICE)
    model.train()
    # Get the model state (train or eval)
    print("Model is in", "train" if model.training else "eval", "mode")
    epochs = 10
    batch_size = 1
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    dataset = TextToSQLDataset(os.path.join("F:", "tfg", "processed", "hf", "sql_context-train.csv"), model.tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        tb = tqdm(dataloader, desc="Batches")
        for inputs, targets in tb:
            print("Inputs:", inputs[0], "Targets:", targets[0])
            for i, t in zip(inputs, targets):
                i, t = i[0].to(DEVICE), t[0].to(DEVICE)
                if DEBUG >= 4: print("I:", i, "T:", t)
                outs: CausalLMOutputWithCrossAttentions = model(i, labels=t)
                print("Outs:", outs)
                logits = outs.logits
                loss = outs.loss
                print("Logits:", logits, "Loss:", loss)
                best = torch.argmax(F.softmax(logits, dim=-1), dim=-1)
                if DEBUG >= 3: print("Logits:", logits, "Target:", t, "Best:", best)
                size = logits.size()
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), t.view(-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tb.set_postfix_str(f"Loss: {loss.item():.4f}")
    print("Done")            
