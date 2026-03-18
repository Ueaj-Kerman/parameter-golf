# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Parameter Golf is an OpenAI Model Craft Challenge: train the best language model that fits in a **16MB artifact** (code + compressed model) and trains in **under 10 minutes on 8xH100s**. Evaluated by compression on FineWeb validation set using **tokenizer-agnostic bits-per-byte (BPB)**.

## Commands

### Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 data/cached_challenge_fineweb.py --variant sp1024          # full dataset (80 shards)
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1  # minimal for smoke tests
```

### Training (PyTorch, 8xH100 via Modal)
```bash
# First run only: download data to Modal volume
modal run modal_train.py::download_data

# Train — ALWAYS give a descriptive RUN_ID (architecture/change/purpose)
RUN_ID=baseline_9L_512d modal run modal_train.py
RUN_ID=wider_12L_768d NUM_LAYERS=12 MODEL_DIM=768 modal run modal_train.py

# Disable wandb for clean timing benchmarks
RUN_ID=bench_baseline WANDB=0 modal run modal_train.py

# Retrieve artifacts
modal volume get parameter-golf-data outputs/final_model.int8.ptz .

# Kill a running Modal app (local ctrl-c does NOT stop remote containers)
modal app list
modal app stop <APP_ID>
```
Hyperparameters are set via environment variables (see `Hyperparameters` class at top of `train_gpt.py`).
Wandb logging is on by default (project: `parameter-golf`). Set `WANDB=0` to disable.

### Training (PyTorch, 8xH100 direct)
```bash
RUN_ID=my_run torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Training (MLX, Apple Silicon)
```bash
RUN_ID=mlx_smoke ITERATIONS=200 python3 train_gpt_mlx.py
```

### Output
- `logs/<RUN_ID>.txt` — training log
- `final_model.pt` — raw state dict
- `final_model.int8.ptz` — quantized + zlib compressed model (submission artifact)

## Architecture

### Model (`train_gpt.py`)
A ~524K parameter GPT with encoder-decoder skip connections:
- **9 transformer layers** split into encoder (0-4) and decoder (5-8) halves
- Decoder layers mix in stored encoder hidden states via learnable `skip_weights`
- **512 dim, 8 attn heads, 4 KV heads** (grouped-query attention)
- **RoPE** positional encoding, **ReLU² MLP** activation, **RMSNorm**
- **Tied embeddings** (input/output share weights), **logit softcap** at 30.0
- `CastedLinear`: stores weights fp32, casts to bfloat16 at compute time

### Three-Optimizer Training
1. **Embedding params** → Adam (LR 0.05)
2. **2D matrix params** in blocks → **Muon** optimizer (LR 0.04) — orthogonalizes gradients via Newton-Schulz iteration
3. **Scalar/vector params** (attn_scale, mlp_scale, resid_mix, skip_weights, q_gain) → Adam (LR 0.04)

### Post-Training Quantization
Per-row int8 quantization + zlib compression to meet the 16MB cap. Small tensors (<65K elements) kept as fp16.

### Validation
Tokenizer-agnostic BPB: counts actual UTF-8 bytes per token via SentencePiece, handles boundary tokens and leading spaces correctly.

## Run Naming Convention
Every training run MUST have a descriptive `RUN_ID` that encodes what changed. Examples:
- `baseline_9L_512d` — baseline config
- `deeper_12L_512d` — more layers
- `wider_9L_768d` — wider model dim
- `muon_lr0.06_9L_512d` — optimizer tweak
- `relu3_9L_512d` — activation change

Never use generic names like `test`, `run1`, `experiment`. The RUN_ID appears in wandb and logs.

## Constraints
- **16MB cap** = 16,000,000 bytes (decimal) for code + compressed model combined
- **10 min wallclock** on 8xH100s for official track
- Single `train_gpt.py` file, max 1500 lines
- No network calls during eval
- Submissions must beat SOTA by ≥0.005 nats (p<0.01)

## Submission Structure
PRs add a dated folder under `records/track_10min_16mb/` containing: `README.md`, `submission.json`, `train.log`, and `train_gpt.py` snapshot.

## Data Format
Binary shards: 256-int header (magic `20240520`, version `1`), tokens as uint16. Tokenizer: SentencePiece with 1024 BPE vocab (`data/tokenizers/fineweb_1024_bpe.model`).
