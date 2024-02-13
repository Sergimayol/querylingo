# pip install -q transformers accelerate
import os, warnings, torch, torch.nn as nn  # noqa: E401
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from utils import Timing, load_safetenors
from safetensors import safe_open

tensors = {}
tensors = load_safetenors("model.safetensors")
with open("keys.txt", "w") as f:
    # Write the keys with the size of the tensor
    f.write("\n".join([f"{key}: {tensors[key].size()}" for key in tensors.keys()]))

warnings.filterwarnings("ignore")
"""
checkpoint = "bigscience/bloomz-560m"
checkpoint = "mistralai/Mistral-7B-Instruct-v0.2"
checkpoint = "bigscience/bloom-3b"
device = "cuda"

model = AutoModelForCausalLM.from_pretrained(checkpoint, cache_dir=os.path.join("D:", "models")).to(device)
tok = AutoTokenizer.from_pretrained(checkpoint)

print(model)
exit()

print("[INFO] Using model:", checkpoint)

while True:
    print("Prompt: ")
    _input = input()
    if _input == "exit":
        break
    with Timing("model.generate: "):
        inputs = tok(_input, return_tensors="pt").to("cuda")
        streamer = TextStreamer(tok)
        _ = model.generate(**inputs, streamer=streamer, temperature=0.9)
"""


class BloomAttention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BloomAttention, self).__init__()
        self.query_key_value = nn.Linear(input_size, 3 * hidden_size, bias=True)
        self.dense = nn.Linear(input_size, hidden_size, bias=True)
        self.attention_dropout = nn.Dropout(p=0.0, inplace=False)

    def forward(self, x):
        qkv = self.query_key_value(x)
        query, key, value = torch.chunk(qkv, 3, dim=-1)
        attention_weights = torch.matmul(query, key.transpose(-2, -1)) / (query.size(-1) ** 0.5)
        attention_weights = torch.nn.functional.softmax(attention_weights, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        output = torch.matmul(attention_weights, value)
        output = self.dense(output)
        return output


class BloomMLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BloomMLP, self).__init__()
        self.dense_h_to_4h = nn.Linear(input_size, 4 * hidden_size, bias=True)
        self.gelu_impl = nn.GELU()
        self.dense_4h_to_h = nn.Linear(4 * hidden_size, hidden_size, bias=True)

    def forward(self, x):
        x = self.dense_h_to_4h(x)
        x = self.gelu_impl(x)
        x = self.dense_4h_to_h(x)
        return x


class BloomBlock(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BloomBlock, self).__init__()
        self.input_layernorm = nn.LayerNorm(input_size, eps=1e-05, elementwise_affine=True)
        self.self_attention = BloomAttention(input_size, hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(input_size, eps=1e-05, elementwise_affine=True)
        self.mlp = BloomMLP(input_size, hidden_size)

    def forward(self, x):
        x = self.input_layernorm(x)
        attention_output = self.self_attention(x)
        x = x + attention_output
        x = self.post_attention_layernorm(x)
        mlp_output = self.mlp(x)
        x = x + mlp_output
        return x


class BloomModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, num_blocks, hidden_size):
        super(BloomModel, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_size)
        self.word_embeddings_layernorm = nn.LayerNorm(embedding_size, eps=1e-05, elementwise_affine=True)
        self.h = nn.ModuleList([BloomBlock(embedding_size, hidden_size) for _ in range(num_blocks)])
        self.ln_f = nn.LayerNorm(embedding_size, eps=1e-05, elementwise_affine=True)

    def forward(self, x):
        x = self.word_embeddings(x)
        x = self.word_embeddings_layernorm(x)
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        return x


class BloomForCausalLM(nn.Module):
    def __init__(self, vocab_size, embedding_size, num_blocks, hidden_size):
        super(BloomForCausalLM, self).__init__()
        self.transformer = BloomModel(vocab_size, embedding_size, num_blocks, hidden_size)
        self.lm_head = nn.Linear(embedding_size, vocab_size, bias=False)

    def forward(self, x):
        x = self.transformer(x)
        x = self.lm_head(x)
        return x


# Definir las dimensiones según tu caso
vocab_size = 250880
embedding_size = 1024
num_blocks = 24
hidden_size = 1024

# Crear la instancia del modelo
model = BloomForCausalLM(vocab_size, embedding_size, num_blocks, hidden_size)
print(model)

state_dict = torch.load("pytorch_model.bin", map_location="cpu")
asd = {"transformer." + key: value for key, value in state_dict.items()}
asd["lm_head.weight"] = torch.rand((vocab_size, embedding_size))
model.load_state_dict(asd)

checkpoint = "bigscience/bloomz-560m"
tok = AutoTokenizer.from_pretrained(checkpoint)

# Forwardear el modelo
inputs = tok("Hello World", return_tensors="pt")
while True:
    output = model(inputs["input_ids"])
    output = torch.nn.functional.softmax(output, dim=-1)
    token = torch.argmax(output)
    print(tok.decode(token.item()), token)
    tmp = inputs["input_ids"][0].tolist()
    tmp.append(token.item())
    inputs = tok(tok.decode(torch.tensor(tmp)), return_tensors="pt")
