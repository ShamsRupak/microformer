"""Model configuration for MicroFormer."""

from __future__ import annotations

from dataclasses import dataclass

_PRESETS: dict[str, dict[str, int | float]] = {
    "gpt2-small": {
        "d_model": 768,
        "n_heads": 12,
        "n_layers": 12,
        "d_ff": 3072,
        "vocab_size": 50257,
        "max_seq_len": 1024,
        "dropout": 0.1,
    },
}


@dataclass(frozen=True)
class ModelConfig:
    """Immutable configuration for a MicroFormer model.

    All architectural hyper-parameters live here so they can be serialised
    alongside a checkpoint without ambiguity.
    """

    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    d_ff: int = 3072
    vocab_size: int = 50257
    max_seq_len: int = 1024
    dropout: float = 0.1
    rope_theta: float = 10_000.0

    # GQA: number of key/value heads.  0 → same as n_heads (standard MHA).
    n_kv_heads: int = 0

    # Derived — set automatically in __post_init__.
    d_head: int = 0

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by "
                f"n_heads ({self.n_heads})"
            )
        # Frozen dataclass requires object.__setattr__ for derived fields.
        object.__setattr__(self, "d_head", self.d_model // self.n_heads)

        # Resolve n_kv_heads: 0 means "same as n_heads" (standard MHA).
        effective_kv = self.n_kv_heads if self.n_kv_heads > 0 else self.n_heads
        object.__setattr__(self, "n_kv_heads", effective_kv)
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by "
                f"n_kv_heads ({self.n_kv_heads})"
            )

    @classmethod
    def from_name(cls, name: str) -> ModelConfig:
        """Instantiate a config from a preset name (e.g. ``'gpt2-small'``)."""
        if name not in _PRESETS:
            raise KeyError(
                f"Unknown preset '{name}'. Available: {list(_PRESETS.keys())}"
            )
        return cls(**_PRESETS[name])
