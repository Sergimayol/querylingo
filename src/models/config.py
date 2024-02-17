from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int
    n_positions: int
    n_ctx: int
    n_embd: int
    n_layer: int
    n_head: int
    eps: float
    initializer_range: float


@dataclass
class GPT2Config(ModelConfig):
    vocab_size: int = 50257
    n_positions: int = 1024
    n_ctx: int = 1024
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    eps: float = 1e-5
    initializer_range: float = 0.02
