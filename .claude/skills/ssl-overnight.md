---
name: ssl-overnight
description: Overnight SSL embedding initialization experiments on local RTX 5090
user_invocable: true
---

# SSL Embedding Initialization — Overnight Experiment Plan

## Setup
- **Hardware**: RTX 5090 (32GB VRAM), single GPU
- **Run command**: `RUN_ID=<name> MAX_WALLCLOCK_SECONDS=<sec> python train_gpt.py` (single GPU, no torchrun)
- **Data**: 1 training shard locally at `data/datasets/fineweb10B_sp1024/`
- **Baseline**: 80-minute single-GPU run with untied 8L at `TIED_EMBED_LR=0.01`

## Core Principle
The SSL prediction target should be **lower dimensional** than the input. A causal conv1d naturally achieves this:
- Input to the conv: `kernel_size * model_dim` elements (from neighboring tokens)
- Output: `model_dim` (predicting next token's input embedding)

This compression forces the conv to learn useful local patterns, which shapes the embeddings to encode meaningful local context.

## Base Config for All Experiments
- `TIE_EMBEDDINGS=0` (untied embeddings — SSL can freely shape input embedding)
- `NUM_LAYERS=8` (drop a layer to fit 16MB with untied)
- `TIED_EMBED_LR=0.01` (our best LR finding)
- SSL runs BEFORE warmup steps (no torch.compile needed)
- After SSL: keep embedding weights + optimizer state, discard conv

## Experiments to Try (each in a separate worktree)

### 1. Simple conv1d, large kernel, 100 steps
- Just a `nn.Conv1d(dim, dim, kernel_size=16, groups=dim)` (depthwise causal)
- Down-project to dim with a linear layer
- Predict next token's input embedding (cosine loss)
- SSL LR: 0.03 for embedding, 1e-3 for conv
- This is the simplest possible version — have we even tried this?

### 2. Simple conv1d, vary kernel size
- Same as #1 but try kernel sizes: 4, 8, 16, 32
- Pick the best, then try more/fewer SSL steps

### 3. Gated conv1d (both up and gate are conv), large kernel
- `conv_up`: Conv1d(dim, hidden=2048, kernel_size=8)
- `conv_gate`: Conv1d(dim, hidden=2048, kernel_size=8)
- SiLU gating, down-project to dim
- Predict next token embedding (cosine loss)
- This was tried on Modal but with bad LR — retry with 0.03

### 4. Multi-step prediction
- Instead of predicting just next token, predict next 2-4 tokens
- Separate prediction heads (or one shared conv with different offsets)
- Loss = average cosine sim across all targets

### 5. Contrastive SSL
- Instead of regression, use contrastive loss
- Positive: next token embedding
- Negatives: random embeddings from the batch
- InfoNCE loss

### 6. Vary SSL steps
- Take best architecture from above
- Try 50, 100, 200, 500 SSL steps
- Find the sweet spot before returns diminish

## How to Run Each Experiment
```bash
# Create worktree
git worktree add -b ssl-<name> ../parameter-golf-ssl-<name>

# Edit train_gpt.py in the worktree to implement the SSL variant

# Run 80-minute local baseline first (if not done)
cd /mnt/c/Users/devse/PycharmProjects/parameter-golf
RUN_ID=local_baseline_untied_8L MAX_WALLCLOCK_SECONDS=4800 \
  TIE_EMBEDDINGS=0 NUM_LAYERS=8 TIED_EMBED_LR=0.01 \
  python train_gpt.py

# Run experiment
cd ../parameter-golf-ssl-<name>
RUN_ID=ssl_<name>_8L MAX_WALLCLOCK_SECONDS=4800 \
  TIE_EMBEDDINGS=0 NUM_LAYERS=8 TIED_EMBED_LR=0.01 \
  python train_gpt.py
```

## Evaluation
Compare final val_bpb against the local baseline. If an experiment shows improvement locally, verify on Modal with 8xH100 at 10 minutes.

## Budget Note
Stop launching new experiments if approaching the $200 sub limit. Prioritize experiments 1-3 first.
