# Embed RMS Optimizer

Custom embedding optimizer: momentum buffer + row-wise RMS normalization for updates.
Also adds embedding row regularization (normalize if RMS ≥ 1.25) and learnable output_scale scalar.

## LR Sweep (3-min, baseline: 1.2619)
| LR | quantized val_bpb |
|----|------|
| 0.001 | 1.2561 |
| 0.003 | 1.2560 |
| 0.005 | **1.2557** |
| 0.01 | 1.2582 |
| 0.03 | 1.2723 |
| 0.05 | 1.2538 (1-min run, not comparable) |
| 0.1 | 1.3204 |
| 0.2 | 1.3401 |
| 0.5 | 1.3448 |

## 10-Minute Results
| Config | quantized val_bpb | val_loss | Δ vs baseline |
|--------|------|---|---|
| LR=0.005 | 1.2204 | 2.0653 | -0.0085 |
| LR=0.01 | 1.2235 | 2.0651 | -0.0054 |
| LR=0.05 (original) | 1.2538 | 2.1067 | +0.0249 |

**Verdict**: Good improvement but superseded by Adam LR=0.01 (1.2175) which is simpler. The improvement was mostly from lower LR, not the optimizer itself.
