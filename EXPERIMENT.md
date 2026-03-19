# Scalar vs Per-Channel Scale

Replaced per-channel attn_scale and mlp_scale (512-element vectors) with single scalars.

## Results (with TIED_EMBED_LR=0.01)
| Config | 10-min quantized val_bpb | Δ |
|--------|------|---|
| Per-channel (baseline + LR fix) | 1.2175 | — |
| Scalar | 1.2189 | +0.0014 |

**Verdict**: Per-channel scales are earning their keep. The 0.0014 difference is ~3x the noise floor. Not worth simplifying.
