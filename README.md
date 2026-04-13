# MicroFormer

MicroFormer is a GPT-style transformer built from scratch in Python/PyTorch with zero HuggingFace model dependencies.
Every component — tokenizer, attention, training loop, inference server — is implemented from first principles.

## Architecture

- RoPE (rotary positional encoding) on Q/K via complex-number rotation
- RMSNorm with float32 upcast for numerical stability
- Weight tying between token embedding and LM head
- Scaled residual projection init (1/√(2n) per GPT-2)
- GELU feedforward with tanh approximation
- Causal self-attention with fused QKV projection and triu mask

## Components

| Component | File | Description | Tests |
|---|---|---|---|
| Model | `microformer/model.py` | GPT-style transformer stack with `generate()` | — |
| Attention variants | `microformer/attention_variants.py` | MHA, MQA, GQA as drop-in modules | 18 |
| BPE Tokenizer | `microformer/tokenizer.py` | Byte-level BPE, no tiktoken dependency | 22 |
| Trainer | `microformer/trainer.py` | AdamW, cosine LR, grad accumulation, checkpointing | 18 |
| Eval | `microformer/eval.py` | Perplexity and BLEU via sacrebleu | 13 |
| Inference server | `microformer/server.py` | FastAPI with KV-cache and batched generation | 19 |
| Ablation | `ablation/run_ablation.py` | MHA vs MQA vs GQA benchmark | — |

112 tests total.

## Ablation Results

Tiny model (4 layers, 256 dim, 8 heads), 500 training steps on random data, CPU.

| Variant | Params | Loss | Perplexity | Throughput (tok/s) |
|---|---|---|---|---|
| MHA | 2,230,528 | 6.2436 | 514.69 | 9,427 |
| MQA | 1,771,776 | 6.2403 | 513.02 | 9,704 |
| GQA | 2,230,528 | 6.2436 | 514.69 | 9,143 |

MQA achieves 20.6% parameter reduction and 2.9% throughput gain vs MHA with equivalent perplexity on this run; KV-cache memory advantage becomes 8x at inference time.

## Quickstart

```
pip install -e ".[dev]"
pytest
MICROFORMER_CHECKPOINT=path/to/ckpt uvicorn microformer.server:app
```

## License

MIT

---

Built from scratch as a learning and portfolio project — every component implemented without high-level ML framework wrappers.
