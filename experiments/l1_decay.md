# L1 Weight Decay + Laplace Initialization

**Branch**: worktree-l1
**Date**: 2026-03-23

## Hypothesis
L1 weight decay (`lr * d * sign(w)`) promotes sparsity and corresponds to a Laplace prior on weights. Changed initializations to match: Laplace distribution with scale `b = 1/sqrt(6*fan_in)` (variance-matched to Kaiming uniform). Truncated variant clips at ±2b to reduce outlier sensitivity.

## Math
- L2 decay ↔ Gaussian prior, L1 decay ↔ Laplace prior
- Laplace(0, b) has variance 2b². Kaiming uniform has variance 1/(3·fan_in)
- Matching: b = 1/√(6·fan_in)
- muP activation scales preserved: CLT ensures output variance matches regardless of weight distribution shape (same fan_in, same variance → same output variance)

## Results (3-minute, 8xH100, quantized val_bpb)

| Run | Init | L1 Decay | val_bpb | Δ vs baseline |
|-----|------|----------|---------|---------------|
| Baseline | Kaiming uniform | 0 | 1.2619 | — |
| l1_decay_laplace_init | Laplace | 0.01 | 1.3315 | +0.0696 |
| l1_decay001_laplace_init | Laplace | 0.001 | 1.3310 | +0.0691 |
| trunc_laplace_no_decay | Truncated Laplace (±2b) | 0 | 1.3407 | +0.0788 |
| normal_init_l1_wd01 | Kaiming uniform | 0.01 | 1.3539 | +0.0920 |

## Analysis
- **Both components hurt independently**: Laplace init alone is +0.079 worse, L1 decay alone is +0.092 worse
- **L1 decay strength barely matters**: 0.01 vs 0.001 gave nearly identical results (both ~+0.07), suggesting the Laplace init was the dominant issue in combined runs
- **Truncation didn't help**: ±2b clipping still produced +0.079 degradation — the issue isn't just tail outliers
- **Early instability**: Step 2 spike to ~17.0 train_loss appeared in ALL runs (including normal init), so this is a model property, not init-related
- **L1 decay is too aggressive for Muon**: Effective per-step decay = lr × d = 0.04 × 0.01 = 0.0004. Even this small amount significantly hurts. Muon's orthogonalized updates may already provide implicit regularization

## Verdict
**Negative result.** Both L1 weight decay and Laplace initialization hurt significantly (~0.07-0.09 BPB). The Laplace distribution's heavier tails (excess kurtosis 3 vs -1.2 for uniform) likely cause worse conditioning despite matched variance. L1 decay conflicts with Muon's orthogonalization-based updates.
