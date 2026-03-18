---
name: modal
description: How to use Modal for training runs in this project
user_invocable: true
---

# Modal Training Infrastructure

## Quick Reference

```bash
# Standard training run (10 min wallclock)
RUN_ID=baseline_9L_512d modal run modal_train.py

# Short iteration run (1 min wallclock)
RUN_ID=test_idea MAX_WALLCLOCK_SECONDS=60 modal run modal_train.py

# Disable wandb for clean timing
RUN_ID=bench WANDB=0 modal run modal_train.py

# Compile-only (warm torch.compile cache, no training)
COMPILE_ONLY=1 modal run modal_train.py

# Retrieve artifacts from volume
modal volume get parameter-golf-data outputs/final_model.int8.ptz .
modal volume get parameter-golf-data outputs/ ./outputs

# Kill a running app (local ctrl-c does NOT stop remote containers)
modal app list
modal app stop <APP_ID>
```

## Hyperparameter Env Vars

All forwarded from local shell to remote container:
`RUN_ID`, `SEED`, `ITERATIONS`, `WARMDOWN_ITERS`, `WARMUP_STEPS`,
`TRAIN_BATCH_TOKENS`, `TRAIN_SEQ_LEN`, `MAX_WALLCLOCK_SECONDS`,
`VOCAB_SIZE`, `NUM_LAYERS`, `NUM_KV_HEADS`, `MODEL_DIM`, `NUM_HEADS`,
`MLP_MULT`, `TIE_EMBEDDINGS`, `ROPE_BASE`, `LOGIT_SOFTCAP`,
`EMBED_LR`, `HEAD_LR`, `TIED_EMBED_LR`, `MATRIX_LR`, `SCALAR_LR`,
`MUON_MOMENTUM`, `VAL_LOSS_EVERY`, `TRAIN_LOG_EVERY`, `WANDB`

## Architecture

- **Image**: CUDA 12.8 devel + all pip packages cached. Rebuilds only when dependencies change.
- **Code**: `train_gpt.py` and `data/` are added to the image via `add_local_file`/`add_local_dir`. Changes trigger a fast layer rebuild (no pip reinstall).
- **Volume** (`parameter-golf-data`): Persistent storage for FineWeb shards, tokenizer, torch inductor cache, and training outputs.
- **wandb**: Logs to project `parameter-golf`. Parses stdout from torchrun subprocess — no wandb code in `train_gpt.py`.
- **Secrets**: `wandb-secret` must exist in Modal with `WANDB_API_KEY`.

## If the volume is lost

Data can be re-downloaded: `modal run modal_train.py::download_data`
This fetches 80 FineWeb shards + tokenizer from HuggingFace (~10GB). Takes a few minutes.
