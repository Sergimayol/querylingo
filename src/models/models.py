"""Models used in the project:
* Under 7b params models:
    - https://huggingface.co/bigscience/bloomz-560m
    - https://github.com/NumbersStationAI/NSQL
    - https://github.com/defog-ai/sqlcoder
    - https://huggingface.co/microsoft/phi-2
    - https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2
    - https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B
    - https://huggingface.co/Deci/DeciLM-7B
    - https://huggingface.co/seeklhy
    - https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2
    - https://huggingface.co/machinists/Mistral-7B-SQL
    - https://huggingface.co/mrm8488/t5-base-finetuned-wikiSQL
    - https://huggingface.co/juierror/text-to-sql-with-table-schema
    - https://huggingface.co/allenai/OLMo-7B
    - https://huggingface.co/allenai/OLMo-1B
    - https://huggingface.co/vikhyatk/moondream1
    - https://huggingface.co/chatdb/natural-sql-7b
    - https://huggingface.co/defog/sqlcoder-7b-2
    - https://huggingface.co/meta-llama/Llama-2-7b
    - https://huggingface.co/meta-llama/Llama-2-7b-chat
    - https://huggingface.co/openlm-research/open_llama_3b
    - https://huggingface.co/bigscience/bloom-3b
    - https://huggingface.co/openai-community/gpt2
    - https://huggingface.co/openai-community/gpt2-large
    - https://huggingface.co/openai-community/gpt2-medium
    - https://huggingface.co/openai-community/gpt2-xl
"""

from typing import Literal
from torch import tril, ones
from torch.nn import Module, Linear, GELU, LayerNorm, Embedding, ModuleList
from transformers import GPT2LMHeadModel, GPT2Tokenizer

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


# GPT2 wrapper from HuggingFace
class GPT2Wrapper(Module):
    def __init__(self, checkpoint: Literal["gpt2", "gpt2-large", "gpt2-medium", "gpt2-xl"] = "gpt2"):
        super(GPT2Wrapper, self).__init__()
        self.model = GPT2LMHeadModel.from_pretrained(checkpoint, cache_dir=CACHE_DIR)
        self.tokenizer = GPT2Tokenizer.from_pretrained(checkpoint)

    def forward(self, input_ids):
        return self.model(input_ids)

    def generate(self, input_ids, attention_mask, max_length=50):
        return self.model.generate(input_ids, attention_mask=attention_mask, max_length=max_length)

    def decode(self, input_ids):
        return self.tokenizer.decode(input_ids, skip_special_tokens=True)

    def encode(self, text):
        return self.tokenizer.encode(text, return_tensors="pt")
