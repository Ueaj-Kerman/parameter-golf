# Adam Embed LR Sweep

**Finding**: Default TIED_EMBED_LR=0.05 is too high. Lower is monotonically better.

## 3-Minute Sweep (baseline: 1.2619 ± 0.0005 quantized val_bpb)
| LR | quantized val_bpb | Δ |
|----|------|---|
| 0.01 | **1.2527** | -0.0092 |
| 0.02 | 1.2567 | -0.0052 |
| 0.03 | 1.2600 | -0.0019 |
| 0.05 | 1.2619 | baseline |
| 0.08 | 1.2706 | +0.0087 |
| 0.10 | 1.2783 | +0.0164 |
| 0.15 | 1.2886 | +0.0267 |

## 10-Minute Full Run (baseline: 1.2289 ± 0.0004)
| LR | quantized val_bpb | val_loss | Δ |
|----|------|---|---|
| **0.01** | **1.2175** | 2.0577 | **-0.0114** |
| 0.05 | 1.2289 | 2.0623 | baseline |

**Verdict**: WINNER. Change default TIED_EMBED_LR from 0.05 to 0.01. Zero code changes needed.
