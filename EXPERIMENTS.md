# Gated Causal Conv Experiments

## Architecture

Replaced the first transformer block with a `GatedCausalConv` module:
- **linear_up**: Linear(512 → H) — standard projection
- **conv_gate**: Conv1d(512 → H, kernel=3, causal) — local context via causal convolution
- **gating**: SwiGLU-style `silu(up) * gate`
- **down_proj**: Linear(H → 512, zero-init)

Conv weights go through Muon optimizer (reshaped from 3D to 2D for Newton-Schulz orthogonalization).

## Key Findings

- Muon is critical for conv weights — Adam performed ~0.075 bpb worse
- Replacing attention+MLP with conv saves ~2ms/step (~4% faster)
- Quality roughly matches baseline but doesn't clearly beat it after quantization

## Results

### 3-Minute Runs

| Config | val_bpb | val_bpb (quant) | steps | step_avg | params |
|--------|---------|-----------------|-------|----------|--------|
| **Baseline** | 1.2581 | **1.2619** | 3911 | 46.0ms | 17.0M |
| Conv h=512 (both conv, sigmoid gate, Adam) | 1.3343 | 1.3414 | 4005 | 44.95ms | 17.1M |
| Conv h=512 (both conv, SwiGLU, Muon) | 1.2599 | 1.2638 | 4081 | 44.12ms | 17.1M |
| Conv h=768 (linear up, conv gate, SwiGLU, Muon) | **1.2550** | **1.2584** | 4096 | 43.95ms | 17.2M |

### 10-Minute Runs

| Config | val_bpb | val_bpb (quant) | steps | step_avg | params | size (int8+zlib) |
|--------|---------|-----------------|-------|----------|--------|------------------|
| **Baseline** | — | **1.2289** | ~13k | ~46ms | 17.0M | — |
| Conv h=768 tied 8blk | 1.2195 | 1.2269 | 13,537 | 44.3ms | 17.2M | 15.83MB |
| Conv h=1280 untied 7blk | 1.2235 | 1.2247 | 14,258 | 42.1ms | 17.2M | 15.99MB |

### Submission Viability

Neither config clears the **0.002 bpb** meaningful improvement threshold over baseline (1.2289 quantized):
- h=768 tied 8blk: 1.2269 quant → Δ = -0.0020 (borderline, within noise)
- h=1280 untied 7blk: 1.2247 quant → Δ = -0.0042 (better, but trades a full block)

## Iteration History

1. **Two conv (sigmoid gate, Adam)** — 1.3414 quant. Very bad. Sigmoid gating and Adam on conv weights both suboptimal.
2. **Two conv (SwiGLU, Muon)** — 1.2638 quant. Muon + SwiGLU recovered most quality. Matched baseline.
3. **Linear up + conv gate (h=768, Muon)** — 1.2584 quant. Wider hidden dim from cheaper linear up. Best 3-min result.
4. **10-min h=768 tied** — 1.2269 quant. Borderline improvement over baseline.
5. **10-min h=1280 untied 7blk** — 1.2247 quant. Traded a block for bigger conv + untied head. Slightly better quantized score but fewer blocks hurt pre-quant quality.

## Conclusions

- The conv layer is a **speed win** (~4% faster per step) but not a clear quality win
- Best used as a building block alongside other improvements, not standalone
- Untying embeddings adds ~480KB compressed, requiring a block reduction to stay under 16MB
- Muon handles 3D conv weights well via reshape to 2D — no need for Adam fallback
