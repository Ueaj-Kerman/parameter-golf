# SSL Embedding Pre-training

100-step SSL stage before warmup using gated causal conv (both up and gate are Conv1d, k=6, hidden=2048) to predict next-token input embedding via cosine similarity loss.

## Timeline

### Attempt 1: Tied embeddings, default LR (ssl_embed_lr_mult=1.0)
Agent implemented:
1. `GatedCausalConv` class — dual Conv1d (up + gate), kernel=6, hidden=2048, plus CastedLinear down projection
2. SSL hyperparameters: ssl_steps=100, ssl_embed_lr_mult=1.0, ssl_conv_lr=1e-3
3. SSL pre-training loop before warmup with optimizer state transfer to main training
4. Environment variable forwarding in modal_train.py

3-minute result:
- SSL loss: 1.0003 -> 0.7207 in 100 steps (~4.9s)
- **Initial val_loss: 32** (vs normal ~6.7) — catastrophic
- Initial train losses: 31.9, 27.7, 22.5... (far worse than normal ~6.9)
- Final quantized val_bpb: **1.3148** (baseline: 1.2619)

Agent's diagnosis: SSL pre-training destroys the tied embedding/output head. With `tied_embed_init_std=0.005`, embeddings start very small intentionally. The SSL training blows them up, and since embeddings are tied with the output head, making them "good for prediction" distorts the output logit computation. Initial val_loss of 32 is catastrophically bad — some tokens get very confident wrong predictions.

### Attempt 2: Tied + norm preservation (embedding row norms clamped after each SSL step)
Agent added per-row norm saving/restoration: save initial embedding row norms, restore them after each SSL step. This constrains SSL to only change embedding **directions**, not magnitudes.

3-minute result:
- SSL loss: 1.0003 -> 0.0377 (step 40) -> 0.2840 (oscillates)
- Initial val_bpb: 3.9597 (much more reasonable, baseline ~4.1)
- Final quantized val_bpb: **1.2701** (baseline: 1.2619, delta: +0.008)

10-minute result:
- val_bpb trajectory: ... 1.2583 (step 10000) -> 1.2506 (step 12000) -> final
- Final quantized val_bpb: **1.2337** (baseline: 1.2244, delta: +0.009)
- 13,166 steps

Agent observed: SSL appears to push embedding directions toward local next-token prediction, which may conflict with what the full transformer needs for global context modeling.

### Attempt 3: Untied 8L, LR=1.8 (embed_lr=0.6 x mult=3.0)
Changes: TIE_EMBEDDINGS=0, NUM_LAYERS=8, removed norm preservation, ssl_embed_lr_mult=3.0.
Agent fixed the SSL LR base to use `embed_lr` (0.6) instead of `tied_embed_lr` (0.05) when untied.

3-minute result:
- SSL effective LR: 0.6 x 3.0 = 1.8 (way too high)
- SSL loss pattern: 0.57 at step 20 -> 0.70 at step 100 (diverging)
- Final quantized val_bpb: **1.2770** (baseline: 1.2619, delta: +0.015)

Agent noted: far enough from baseline that 10-minute run unlikely to be competitive.

### Attempt 4: Untied 8L, LR=0.03 (mult=0.05)
Changed ssl_embed_lr_mult from 3.0 to 0.05, giving effective LR of 0.6 x 0.05 = 0.03.

3-minute result:
- SSL loss converged smoothly: 1.0001 -> 0.7087
- Final quantized val_bpb: **1.2609** (baseline: 1.2619, delta: -0.001, marginal improvement)

10-minute result:
- Run was preempted once by Modal and restarted from scratch
- val_bpb trajectory: 1.2706 (step 7000) -> 1.2555 (step 11000) -> final
- Final quantized val_bpb: **1.2299** (unquantized: 1.2281)
- Effectively same as untied 8L without SSL (1.2285)

## Summary
| Run | Duration | Quantized val_bpb | Steps | Notes |
|---|---|---|---|---|
| Baseline | 3 min | **1.2619** | ~3775 | |
| SSL tied, no norm pres | 3 min | 1.3148 | 3775 | Initial val_loss: 32 |
| SSL tied, norm preserved | 3 min | 1.2701 | 3951 | |
| SSL untied 8L, LR=1.8 | 3 min | 1.2770 | — | SSL diverged |
| SSL untied 8L, LR=0.03 | 3 min | **1.2609** | — | Best SSL result |
| Baseline | 10 min | **1.2244** | ~13100 | |
| SSL tied, norm preserved | 10 min | 1.2337 | 13166 | |
| SSL untied 8L, LR=0.03 | 10 min | 1.2299 | — | Same as no-SSL untied 8L |

**Verdict**: SSL with tied embeddings is fundamentally broken (corrupts output head). With untied embeddings + dropped layer, the best SSL result (1.2299) matches untied 8L without SSL (1.2285). Any SSL benefit is eaten by the layer loss. Needs a way to untie without losing a layer.
