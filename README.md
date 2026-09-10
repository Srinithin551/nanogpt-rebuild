# nanoGPT (rebuild)

A from-scratch reimplementation of [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT),
built file by file for learning. The reference copy lives at `../original`.

## Architecture

```
raw text ──prepare.py──> train.bin / val.bin (uint16 token stream) + meta.pkl
                               │
                        train.py (training loop) ──> out_dir/ckpt.pt
                               │  uses                     │
                          model.py (GPT)              sample.py ──> generated text
```

Everything is plain PyTorch — no framework abstractions. The whole project is ~5 files.

## Files

| File | Role |
|------|------|
| `data/shakespeare_char/prepare.py` | Turn raw text into `train.bin`/`val.bin` (char-level) + `meta.pkl` |
| `model.py` | The GPT: `LayerNorm`, `CausalSelfAttention`, `MLP`, `Block`, `GPTConfig`, `GPT` |
| `train.py` | Training loop: data loader, cosine LR schedule, grad accumulation, checkpointing, DDP |
| `configurator.py` | Minimal config system (exec a config file + `--key=value` overrides) |
| `sample.py` | Load a checkpoint and autoregressively generate text |
| `config/train_shakespeare_char.py` | Hyperparameters for the small demo model |

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch numpy tiktoken
```

## Quick start (character-level Shakespeare)

```bash
# 1. prepare the data
python data/shakespeare_char/prepare.py

# 2. train (Apple Silicon shown; use --device=cuda on an NVIDIA GPU, --device=cpu otherwise)
python train.py config/train_shakespeare_char.py --device=mps --compile=False

# 3. sample from the trained model
python sample.py --out_dir=out-shakespeare-char --device=mps
```

`--compile=False` is used because `torch.compile` is not supported on MPS/CPU;
drop it (or set `--compile=True`) on CUDA for a noticeable speedup.

## Key ideas worth remembering

- **A dataset = a flat `uint16` array of token ids + a `meta.pkl`.** That single
  contract is why the trainer is dataset-agnostic.
- **Pre-norm transformer blocks:** `x = x + attn(ln(x))` then `x = x + mlp(ln(x))`.
- **Weight tying:** the token-embedding matrix is reused as the output projection.
- **Gradient accumulation** simulates a large batch by summing grads over several
  micro-batches before one optimizer step.
- **Cosine LR schedule with linear warmup**, and weight decay only on 2D tensors.
