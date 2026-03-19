# Orthogonal Init via Newton-Schulz

Applied zeropower_via_newtonschulz5 to initialize Q/K/V projections and MLP up-projection (fc) to orthogonal matrices.

## Results
- 3-min: comparable to baseline
- 10-min quantized val_bpb: 1.2301 (val_loss: 2.0639)
- Baseline: 1.2289 (val_loss: 2.0623)
- Δ: +0.0012 (within noise)

Pre-quantization was slightly better (1.2223 vs 1.2244) but quantization gap widened.
Model init takes ~44-52s on CPU due to Newton-Schulz iterations.

**Verdict**: No effect. Quantization negates any marginal pre-quant benefit.
