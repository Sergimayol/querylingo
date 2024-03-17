from typing import Dict, Literal
from torch import tril, ones
from torch.nn import Module, Linear, GELU, LayerNorm, Embedding, ModuleList
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, MistralForCausalLM

from .config import ModelConfig
from utils import CACHE_DIR


class MLP(Module):
    def __init__(self, n_state: int, n_embd: int):
        super(MLP, self).__init__()
        # TODO: See why the orginal code uses Conv1D instead of Linear
        self.c_fc = Linear(n_state, n_embd)
        self.act = GELU()
        self.c_proj = Linear(n_embd, n_state)


class MultiHeadAttention(Module):
    def __init__(self, n_embd: int, n_ctx: int, n_head: int, scale: bool = False):
        super(MultiHeadAttention, self).__init__()
        self.c_attn = Linear(n_embd * 3, n_embd)
        self.c_proj = Linear(n_embd, n_embd)
        self.bias = tril(ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx)  # Mask for the attention mechanism
        self.scale = scale
        self.n_head = n_head


class TransformerBlock(Module):
    def __init__(self, n_embd: int, eps: float, n_ctx: int, n_head: int, scale: bool = False):
        super(TransformerBlock, self).__init__()
        self.ln_1 = LayerNorm(n_embd, eps=eps)
        self.attn = MultiHeadAttention(n_embd, n_ctx, n_head, scale)
        self.ln_2 = LayerNorm(n_embd, eps=eps)
        self.mlp = MLP(4 * n_embd, n_embd)


class Transformer(Module):
    def __init__(self, config: ModelConfig):
        super(Transformer, self).__init__()
        self.n_layer = config.n_layer
        self.n_embd = config.n_embd
        self.wpe = Embedding(config.n_positions, self.n_embd)
        self.wte = Embedding(config.vocab_size, self.n_embd)
        self.h = ModuleList([TransformerBlock(self.n_embd, config.eps, config.n_ctx, config.n_head, scale=True) for _ in range(self.n_layer)])
        self.ln_f = LayerNorm(self.n_embd, eps=config.eps)


class GPT2Wrapper(Module):
    def __init__(self, checkpoint: Literal["gpt2", "gpt2-large", "gpt2-medium", "gpt2-xl"] = "gpt2"):
        super(GPT2Wrapper, self).__init__()
        self.model = GPT2LMHeadModel.from_pretrained(checkpoint, cache_dir=CACHE_DIR)
        self.tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained(checkpoint, cache_dir=CACHE_DIR, use_fast=True)

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

    def generate(self, input_ids, attention_mask, max_length=50):
        # TODO: Change this to a custom implementation
        return self.model.generate(input_ids, attention_mask=attention_mask, max_length=max_length)

    def decode(self, input_ids):
        return self.tokenizer.decode(input_ids, skip_special_tokens=True)

    def encode(self, text) -> Dict:
        return self.tokenizer.encode(text, return_tensors="pt")

    def max_length(self) -> int:
        return self.model.config.n_positions


class MistralWrapper(Module):
    def __init__(self, checkpoint: Literal["Mistral-7B-v0.1", "Mistral-7B-Instruct-v0.1", "Mistral-7B-Instruct-v0.2"] = "Mistral-7B-v0.1"):
        super(MistralWrapper, self).__init__()
        checkpoint = f"mistralai/{checkpoint}"
        self.model = MistralForCausalLM.from_pretrained(checkpoint, cache_dir=CACHE_DIR)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

    def generate(self, input_ids, attention_mask, max_length=50):
        return self.model.generate(input_ids, attention_mask=attention_mask, max_length=max_length)

    def decode(self, input_ids):
        return self.tokenizer.decode(input_ids, skip_special_tokens=True)

    def encode(self, text) -> Dict:
        return self.tokenizer.encode(text, return_tensors="pt")
