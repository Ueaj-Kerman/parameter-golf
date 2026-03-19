# Orthogonal Init via Newton-Schulz

Applied `zeropower_via_newtonschulz5` (already in the codebase for Muon) to initialize Q/K/V projections and MLP up-projection (fc) to orthogonal matrices at model creation time.

## Timeline

### Implementation
Agent tagged layers with `_ortho_init = True`:
- `CausalSelfAttention.c_q` (Q projection)
- `CausalSelfAttention.c_k` (K projection)
- `CausalSelfAttention.c_v` (V projection)
- `MLP.fc` (MLP up-projection)

In `_init_weights`, after zero-init processing, calls `zeropower_via_newtonschulz5(module.weight, steps=10)` on tagged modules. No gate projection exists (ReLU^2 MLP has no gate).

Agent noted the function runs before `main()` compiles it, so it uses the raw uncompiled Python version. Also verified the function handles weight shapes correctly — `nn.Linear.weight` is always 2D, and the function handles rows > cols by transposing internally.

### 1-Minute Smoke Test
- No errors or NaN
- Loss: 6.94 -> ~2.35 by step 1200, val_bpb 1.3364 at step 1303
- Model size: 13,478,464 bytes (under cap)
- **Model init took 52.6s** (vs typical few ms) due to Newton-Schulz iterations running in bfloat16 on CPU before `.to(device)`. Significant overhead but one-time cost within the 10-minute budget.

### 10-Minute Run
Training proceeded normally after the long init. val_bpb trajectory: 1.3253 (step 2000) -> 1.2585 (step 8000) -> final.

- Reached step 12,488 in 600 seconds
- Artifact size: 15,869,927 bytes (under 16MB cap)

## Results
| Metric | Ortho Init | Baseline |
|--------|-----------|----------|
| val_bpb (pre-quant) | **1.2223** | 1.2244 |
| val_bpb (post-quant) | 1.2301 | 1.2289 |
| val_loss (pre-quant) | 2.0639 | 2.0623 |
| Quant gap | 0.0078 | ~0.0045 |

- Pre-quantization val_bpb slightly better (1.2223 vs 1.2244)
- Post-quantization val_bpb slightly worse (1.2301 vs 1.2289) due to wider quantization gap
- Model init takes ~44-52s on CPU due to Newton-Schulz iterations

**Verdict**: No meaningful effect. Quantization negates the marginal pre-quant benefit. The 44-52s init overhead is also undesirable.
