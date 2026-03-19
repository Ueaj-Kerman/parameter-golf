# Experiment Log

All experiments run on 2026-03-18/19, comparing against the 9L/512d tied-embeddings baseline.

## Baselines

### 3-Minute Baselines (8xH100, quantized val_bpb)
| Seed | val_bpb | Steps | step_avg |
|------|---------|-------|----------|
| 1337 | 1.2614  | 3911  | 46.0ms   |
| 42   | 1.2624  | 3954  | 45.5ms   |
| 7    | 1.2618  | 3951  | 45.5ms   |
| **Mean ± Std** | **1.2619 ± 0.0005** | | |

### 10-Minute Baselines (8xH100, quantized val_bpb)
| Seed | val_bpb | val_loss | Steps | step_avg |
|------|---------|----------|-------|----------|
| 1337 | 1.2287  | 2.0623   | 12527 | 47.9ms   |
| 42   | 1.2293  | 2.0649   | 13112 | 45.8ms   |
| 7    | 1.2286  | 2.0636   | 12527 | 47.9ms   |
| **Mean ± Std** | **1.2289 ± 0.0004** | | | |

---

## All Results (10-minute, 8xH100, sorted by quantized val_bpb)

| # | Experiment | Branch | quantized val_bpb | Δ vs baseline | Steps | Size | Verdict |
|---|-----------|--------|-------------------|---------------|-------|------|---------|
| 13 | Untied embeds 9L | untied-embeds | 1.2168 | -0.0121 | 13012 | **16.3MB** | **INVALID (over 16MB cap)** |
| — | | | | | | | |
| 1 | **Adam embed LR=0.01** | adam-embed-lr-sweep | **1.2175** | **-0.0114** | 13112 | 15.9MB | **WINNER — zero code changes** |
| 14 | Scalar scale (+ LR=0.01) | scalar-scale | 1.2189 | -0.0100 | 13244 | 15.9MB | Per-channel is better by 0.0014 |
| 2 | Embed RMS opt LR=0.005 | embed-optimizer | 1.2204 | -0.0085 | 13224 | 15.8MB | Good but Adam LR fix is simpler |
| 3 | Embed RMS opt LR=0.01 | embed-optimizer | 1.2235 | -0.0054 | 13020 | 15.8MB | Superseded by LR=0.005 |
| 15 | GatedCausalConv h=1280 untied 7blk | worktree-ssl | 1.2247 | -0.0042 | 14258 | 15.99MB | Best conv config, ~4% faster/step |
| 16 | GatedCausalConv h=768 tied 8blk | worktree-ssl | 1.2269 | -0.0020 | 13537 | 15.83MB | Borderline, within noise |
| 4 | Canon v3 (matmul, LR=0.005) | canon-layers | 1.2276 | -0.0013 | ~10900 | 15.9MB | Marginal, 55ms/step overhead |
| 5 | Untied embeds 8L | untied-embeds | 1.2285 | -0.0004 | 14659 | 14.6MB | Noise, layer loss cancels gain |
| — | **Baseline (seed 7)** | main | **1.2286** | — | 12527 | 15.9MB | |
| — | **Baseline (seed 1337)** | main | **1.2287** | — | 12527 | 15.9MB | |
| — | **Baseline mean ± std** | main | **1.2289 ± 0.0004** | — | | | |
| — | **Baseline (seed 42)** | main | **1.2293** | — | 13112 | 15.9MB | |
| 8 | SSL untied 8L (LR=0.03) | ssl-pretrain | 1.2299 | +0.0010 | 14549 | 14.6MB | Noise |
| 6 | Ortho-init (NS on Q/K/V/fc) | ortho-init | 1.2301 | +0.0012 | 12488 | 15.9MB | No effect |
| 7 | SSL pretrain (tied, norm pres) | ssl-pretrain | 1.2337 | +0.0048 | 13166 | 15.8MB | Hurts with tied embeds |
| 10 | Canon v2 (post-act, k8, conv1d) | canon-layers | 1.2399 | +0.0110 | 7499 | 15.9MB | 78ms/step killed it |
| 12 | Embed RMS opt LR=0.05 | embed-optimizer | 1.2538 | +0.0249 | 13020 | 15.8MB | Default LR too high |
| 9 | SSL untied 8L (LR=1.8, diverged) | ssl-pretrain | 1.2770 | +0.0481 | — | 14.6MB | LR way too high |
| 11 | Canon v1 (pre-act, k4) | canon-layers | 1.2840 | +0.0551 | 2416 | 15.5MB | 75ms/step, wrong placement |
| — | | | | | | | |
| 17 | NorMuon (per-row 2nd moment norm) | worktree-normuon | — | — | — | — | Not run (code-only) |
| 18 | SPlus optimizer (SVD eigenbasis) | worktree-svdopt | — | — | — | — | Not run (code-only) |

---

## Detailed Experiment Notes

### 1. Adam Embed LR Sweep (adam-embed-lr-sweep)
**Finding**: The default `TIED_EMBED_LR=0.05` is too high. Monotonic improvement with lower LR.

3-minute sweep:

| LR | quantized val_bpb |
|----|-------------------|
| 0.01 | **1.2527** |
| 0.02 | 1.2567 |
| 0.03 | 1.2600 |
| 0.05 | 1.2619 (baseline) |
| 0.08 | 1.2706 |
| 0.10 | 1.2783 |
| 0.15 | 1.2886 |

10-min: LR=0.01 → **1.2175** (−0.0114). Zero code changes, just default LR.

### 2. Embed RMS Optimizer (embed-optimizer)
Custom optimizer: momentum buffer + row-wise RMS normalization for updates. Also adds embedding row regularization (normalize if RMS ≥ 1.25) and a learnable output_scale scalar.

LR sweep:

| LR | 3-min quantized |
|----|-----------------|
| 0.001 | 1.2561 |
| 0.003 | 1.2560 |
| 0.005 | **1.2557** |
| 0.01  | 1.2582 |
| 0.03  | 1.2723 |
| 0.1   | 1.3204 |
| 0.2   | 1.3401 |
| 0.5   | 1.3448 |

Plateau at 0.001-0.005. 10-min best (LR=0.005): **1.2204**. But Adam at LR=0.01 beats it (1.2175), meaning the improvement was mostly from lower LR, not the optimizer.

### 3. Canon Layers (canon-layers)
Depthwise causal conv1d per block (post-activation, skip connection, zero-init).

Three iterations:
- **v1**: Pre-activation, nn.Conv1d, k=4. 75-78ms/step (2x baseline). Bad.
- **v2**: Post-activation, nn.Conv1d, k=8, zero-init. Still 78ms/step. transpose-conv-transpose pattern breaks torch.compile fusion.
- **v3**: Post-activation, pure matmul (shifted views), k=4, CANON_LR=0.005. 55ms/step (fixed!). 1.2276 at 10-min, marginal -0.0013 improvement.

Key learning: nn.Conv1d with groups=dim is toxic to torch.compile. Must use manual matmul.

### 4. Untied Embeddings (untied-embeds)
Separate input embedding and zero-initialized output head. Adds 524K params (vocab×dim).

- 9L untied: 1.2168 but 16.3MB — **over 16MB cap**
- 8L untied: 1.2285 — matches 9L tied baseline, layer loss cancels gain
- Fast per-step (40.9ms vs 47.9ms) due to fewer layers

### 5. Orthogonal Init (ortho-init)
Newton-Schulz orthogonalization of Q/K/V projections and MLP up-projection at init.

Result: 1.2301, essentially identical to baseline. Pre-quantization was slightly better (1.2223 vs 1.2244) but quantization gap widened. No benefit.

### 6. SSL Embedding Pre-training (ssl-pretrain)
100-step SSL stage before warmup using gated causal conv (both up and gate are Conv1d, k=6, hidden=2048) to predict next-token input embedding.

Issues encountered:
- **Tied embeds + SSL**: Catastrophic — SSL pushes embeddings for local prediction, destroying the output head. Initial val_loss 32 vs normal 6.
- **Norm preservation hack**: Partially fixed tied version (1.2337) but still worse than baseline.
- **Untied 8L + high LR (1.8)**: SSL loss diverged (0.57→0.70).
- **Untied 8L + low LR (0.03)**: SSL converged (1.0→0.71). 3-min was marginally better (1.2609 vs 1.2619) but 10-min washed out to 1.2299 (noise).

Key learning: SSL on embeddings needs untied weights and conservative LR. May need more steps or different architecture to show benefit.

### 7. Scalar vs Per-Channel Scale (scalar-scale)
Replaced per-channel `attn_scale` and `mlp_scale` (512-element vectors) with single scalars.

With TIED_EMBED_LR=0.01:
- Per-channel: 1.2175
- Scalar: 1.2189
- Δ: +0.0014 (~3x noise floor)

Per-channel scales are earning their keep at this model size.

### 8. GatedCausalConv (worktree-ssl)
Replaced first transformer block with a `GatedCausalConv` module: linear up-proj + causal Conv1d gate + SwiGLU gating + zero-init down-proj. Conv weights handled by Muon (reshaped 3D to 2D for Newton-Schulz).

Key findings:
- Muon is critical for conv weights -- Adam performed ~0.075 bpb worse
- Conv layer saves ~2ms/step (~4% faster) vs attention+MLP block
- Best 3-min: h=768 → 1.2584 quant bpb (baseline 1.2619)
- Best 10-min: h=1280 untied 7blk → **1.2247** quant bpb (baseline 1.2289, Δ=-0.0042)
- h=768 tied 8blk: 1.2269 quant (Δ=-0.0020, borderline)
- wandb runs: `conv_h1280_untied_7blk_full` (val_loss 2.1038), `gated_conv_linup_h768_nosc_muon_8L_512d_full` (val_loss 2.0765)

Iteration history: sigmoid gate+Adam (bad) → SwiGLU+Muon (matched baseline) → linear up+conv gate h=768 (best 3-min) → h=1280 untied (best 10-min). Speed win but not a clear standalone quality win.

### 9. NorMuon -- Per-Row Second Moment Normalization (worktree-normuon)
Modified Muon optimizer to add per-row second moment normalization after Newton-Schulz orthogonalization. Added `beta2` (default 0.95) and `eps` (default 1e-10) hyperparameters.

Algorithm change in Muon step:
1. After Newton-Schulz orthogonalization, compute per-row mean of squared values
2. Track EMA of squared values via `lerp_` with `beta2`
3. Normalize each row by `sqrt(second_moment)`
4. Rescale to preserve overall gradient norm (ratio of pre/post normalization norms)

Replaces the original `max(1, rows/cols)**0.5` scale correction. Idea: adaptively normalize per-neuron update magnitudes while keeping the orthogonalized direction. No training runs recorded.

### 10. SPlus Optimizer -- SVD Eigenbasis (worktree-svdopt)
Replaced Muon entirely with SPlus (Frans, Levine, Abbeel 2025), a stable whitening optimizer that projects momentum into a periodically-recomputed eigenbasis, applies sign updates, and projects back.

Key changes:
- Tracks left and right covariance matrices (`g @ g.T`, `g.T @ g`) with EMA (`b2=0.999`)
- Periodic eigendecomposition via `torch.linalg.eigh` every `inverse_every=100` steps
- Projects momentum into eigenbasis, applies `sign()`, projects back
- Shape-aware LR scaling: `lr * 2/(m+n)` for 2D params, `lr * 0.001` for scalars
- EMA weight averaging with `eval_mode()`/`train_mode()` for validation
- Removed Newton-Schulz iteration entirely, removed `torch.compile` of NS function
- Default LR raised to 0.2 (from Muon's 0.04), removed momentum warmup

No training runs recorded. Large refactor (107 insertions, 86 deletions).

---

## Infrastructure Notes

### torch.compile Startup
- Graph tracing + Triton compilation: ~88-111s on first warmup step
- TORCHINDUCTOR_CACHE_DIR on Modal volume saves ~30s (caches Triton kernels, not graph tracing)
- Total startup overhead: ~164s cached, ~195s cold

### NCCL Hangs
- Encountered 480s NCCL watchdog hangs on Modal (infra issue, not code)
- Fixed by setting `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=60` and `NCCL_TIMEOUT=60`

### Modal Step Speed
- Baseline: ~46ms/step on 8xH100
- Canon with nn.Conv1d: ~78ms/step (transpose-conv-transpose breaks torch.compile)
- Canon with matmul: ~55ms/step (manual shifted views, partially fused)
- Untied 8L: ~41ms/step (fewer layers)
