# $env:DEBUG=3; $env:CACHE_DIR="D:models"; $env:DEVICE="cuda"; python .\src\train.py
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from utils import DEBUG, DEVICE
from models import GPT2Wrapper
from data import TextToSQLDataset


if __name__ == "__main__":
    if DEBUG >= 2: print("DEVICE:", DEVICE)
    model = GPT2Wrapper().to(DEVICE)
    # Get the model state (train or eval)
    print("Model is in", "train" if model.training else "eval", "mode")
    epochs = 10
    batch_size = 256
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        t = tqdm(range(batch_size), desc="Batches")
        # TODO: DELETE ALL THIS (FOR THE MOMENT, IT'S JUST A PLACEHOLDER)
        for batch in t:
            # Forward pass
            outs: CausalLMOutputWithCrossAttentions = model(torch.randint(0, 100, (1, 10)).to(DEVICE))
            # Outputs: class -> CausalLMOutputWithCrossAttentions
            logits = outs.logits
            # Loss
            size = logits.size()
            rand_target = torch.randint(0, 100, (size[0] * size[1],)).to(DEVICE)
            loss = F.cross_entropy(logits.view(size[0] * size[1], -1), rand_target)
            loss_ = F.cross_entropy(logits.view(-1, logits.size(-1)), rand_target)
            assert loss == loss_, f"{loss} != {loss_}"
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t.set_postfix_str(f"Loss: {loss.item():.4f}")
    print("Done")
