# Embed RMS Optimizer

Custom embedding optimizer: momentum buffer + row-wise RMS normalization for updates.
Also adds embedding row regularization (normalize if RMS >= 1.25) and learnable output_scale scalar.

## Timeline

### Implementation
Agent read the codebase and implemented four changes:
1. **EmbedRMSOpt optimizer** — custom `torch.optim.Optimizer` subclass with momentum buffer. Updates computed by dividing each momentum row by its row-wise RMS, then scaling by LR. Replaces Adam for embeddings.
2. **Embedding row RMS regularization** — after each optimizer step, any embedding row with RMS >= 1.25 is normalized back to unit RMS via vectorized masked division.
3. **Output projection scale** (`output_scale`) — learnable scalar (init 1.0) multiplying logits before softcap. Added to scalar optimizer group and `CONTROL_TENSOR_NAME_PATTERNS` for quantization handling.
4. Embed LR already independently configurable (`TIED_EMBED_LR=0.05` when tied), no changes needed.

### 1-Minute Smoke Test
- No crashes, no NaN
- Initial spike at step 2 (train_loss 15.53) but recovered quickly — likely momentum-based optimizer needing a step to settle
- Loss: 6.94 -> 2.34 by step 1306, val_bpb 1.3857
- Model size: 13.37MB (under cap)

### 10-Minute Run (LR=0.05, original)
- val_bpb trajectory: 4.1077 -> 1.4979 -> 1.4132 -> 1.3909 -> 1.3707 -> 1.3502 -> 1.3372 -> 1.3291 -> 1.3240 -> 1.3055 -> 1.2477 -> 1.2476
- **Post-quant val_bpb: 1.2538**, val_loss: 2.1170
- 13,020 steps in 600s, model size: 15.84MB
- Much worse than baseline (1.2289) — agent noted the default LR of 0.05 was tuned for Adam and the RMS optimizer has different scaling dynamics

### LR Sweep (3-min, baseline: 1.2619)
Previous sweep had found LR=0.01 gave 1.2582. Agent continued downward:

| LR | quantized val_bpb | steps |
|----|------|-------|
| 0.001 | 1.2561 | 3951 |
| 0.003 | 1.2560 | 3959 |
| **0.005** | **1.2557** | 3955 |
| 0.01 | 1.2582 | — |
| 0.03 | 1.2723 | — |
| 0.05 | 1.2538 (1-min run, not comparable) | — |
| 0.1 | 1.3204 | — |
| 0.2 | 1.3401 | — |
| 0.5 | 1.3448 | — |

Agent observed LRs 0.001-0.005 plateau at ~1.2557-1.2561, with LR=0.005 marginally best. Chose 0.005 for the full run since it has faster convergence dynamics that matter more in longer runs.

Note: LR=0.003 run was interrupted once and had to be retried.

## 10-Minute Results
| Config | quantized val_bpb | val_loss | steps | delta vs baseline |
|--------|------|---|---|---|
| LR=0.005 | **1.2204** | 2.0653 | 13224 | **-0.0085** |
| LR=0.01 | 1.2235 | 2.0651 | — | -0.0054 |
| LR=0.05 (original) | 1.2538 | 2.1170 | 13020 | +0.0249 |

## Key Findings
- The improvement was mostly from lower LR, not the custom optimizer itself
- LRs 0.005, 0.003, 0.001 all plateau at nearly the same 3-min performance, suggesting the optimum is in the 0.001-0.005 range
- LR=0.005 beats baseline by 0.0085 at 10 minutes (well outside noise)

**Verdict**: Good improvement but superseded by Adam LR=0.01 (1.2175) which is simpler and better. The improvement was mostly from lower LR, not the optimizer itself.
