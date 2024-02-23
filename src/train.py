# $env:DEBUG=3; $env:CACHE_DIR="D:models"; $env:DEVICE="cuda"; python .\src\train.py
from models import GPT2Wrapper
from tqdm import tqdm
from utils import DEBUG, DEVICE
import torch


if __name__ == "__main__":
    model = GPT2Wrapper().to(DEVICE)
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
            outputs = model(torch.randint(0, 100, (1, 10)).to(DEVICE))
            # Outputs: class -> CausalLMOutputWithCrossAttentions
            outputs = outputs.logits
            # Loss
            size = outputs.size()
            loss = torch.nn.functional.cross_entropy(outputs.view(size[0] * size[1], -1), torch.randint(0, 100, (size[0] * size[1],)).to(DEVICE))
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t.set_postfix_str(f"Loss: {loss.item():.4f}")
    print("Done")
