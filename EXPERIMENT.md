# Untied Embeddings with Zero-Init Head

Separate input embedding and output projection. Output head zero-initialized.

## Results
| Config | 10-min quantized val_bpb | val_loss | Size | Δ |
|--------|------|---|---|---|
| 9L untied | 1.2168 | 2.0522 | **16.3MB (OVER CAP)** | -0.0121 |
| 8L untied | 1.2285 | 2.1113 | 14.6MB | -0.0004 |
| 9L tied (baseline) | 1.2289 | 2.0623 | 15.9MB | — |

- 9L untied shows real potential (-0.012) but exceeds 16MB cap by 308KB
- 8L untied fits (14.6MB) but layer loss cancels untying gain
- Faster per-step (40.9ms vs 47.9ms) due to fewer layers, gets more steps (14659 vs 12527)
- LR config already existed: embed_lr=0.6, head_lr=0.008

**Verdict**: Net neutral at 8L. Need compression improvements to fit 9L untied under cap.
