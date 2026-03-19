# Canon Layers (Depthwise Causal Conv1d)

From "Physics of Language Models: Part 4.1" (Allen-Zhu, NeurIPS 2025). Depthwise causal conv1d per block with skip connection and zero-init.

## Research Phase
Agent researched the paper and implementation. Key findings:
- Canon layers are 1D depthwise causal convolutions: `canon(h_t) = h_t + conv1d([h_t, h_{t-1}, ..., h_{t-K+1}])`
- Four insertion points: Canon-A (before attention), Canon-B (on Q/K/V), Canon-C (before MLP), Canon-D (inside MLP)
- Canon-AC adds ~37K params across 9 layers (negligible vs ~17M total)
- Paper validates at 1.3B params; benefit uncertain at 17M scale
- Reference impl: `facebookresearch/PhysicsLM4`, `ShortConvolution` module wrapping `nn.Conv1d`

## Iterations

### v1: Pre-activation, nn.Conv1d, k=4, random init
Agent implemented Canon-A + Canon-C using `nn.Conv1d(dim, dim, kernel_size=4, groups=dim, padding=3)` applied via `conv(x.transpose(1,2))[..., :x.size(1)].transpose(1,2)`.

Key observations during implementation:
- Conv1d weights are 3D (dim, 1, K), which fell outside both `matrix_params` (ndim==2) and `scalar_params` (ndim<2). Agent fixed `scalar_params` to use `p.ndim != 2`.
- Fused Adam requires same dtype in a group. Conv weights (bf16 from `.bfloat16()`) would conflict with fp32 control tensors. Agent added conv weights to fp32 restore logic.
- Model had some confusion about app management — multiple Modal apps were running simultaneously and had to be stopped.

Results:
- **75-78ms/step** (baseline: 46ms) — nearly 2x slowdown
- 3-min: **1.2840** (random init hurts early training)
- Agent identified: random init means conv adds noise at start, should zero-init

Agent zeroed the conv weights for identity-at-init behavior:
- 3-min: **1.2819** (still worse, slight improvement from zero-init)

Note: The train_gpt.py was being modified externally during this phase, causing confusion about which version was running. The file changed between pre/post-activation placement, kernel sizes (4 vs 8), and other details across runs.

### v2: Post-activation, nn.Conv1d, k=8, zero-init
External modifications changed the implementation to post-attention/post-MLP, kernel_size=8, with `F.pad` for causal padding.
- Still **~78ms/step** — the transpose-conv-transpose pattern is the bottleneck
- 10-min: **1.2399** (but this result may be from a different variant due to concurrent edits)
- One run failed with NCCL/NVSwitch hardware errors (Modal infrastructure)

### Speed Investigation
Dedicated agent analyzed the 2x slowdown. Root causes identified:
1. **Transpose-conv-slice-transpose pattern** (18x per step): creates non-contiguous memory accesses, each materializes intermediate tensors that torch.compile cannot fuse through the Conv1d boundary
2. **Dynamic trim `[..., :x.size(1)]`**: may prevent operator fusion even with `dynamic=False`
3. **cuDNN dispatch overhead**: 18 separate small conv1d kernel launches add latency
4. **Conv1d with groups=dim**: may fall back to generic (slow) cuDNN algorithm rather than specialized depthwise kernel

Recommended fix: replace conv1d entirely with pure tensor ops (shifted views + weighted sum) that torch.compile can fully fuse.

### v3: Post-activation, pure matmul (shifted views), k=4, CANON_LR=0.005
Major rewrite: replaced `nn.Conv1d` with raw 2D parameter weights `(dim, K)` and manual shift-multiply-sum. Agent:
- Used `F.pad` on time dimension + unrolled sum of shifted views weighted by per-channel kernel weights
- Stored canon weights as 2D `nn.Parameter` (dim, 4) initialized to zeros
- Added separate `CANON_LR=0.005` with its own Adam optimizer group
- Fixed optimizer split: excluded `canon_` params from `matrix_params` (they're 2D but shouldn't go to Muon)
- Fixed double computation of `self.attn_norm(x)` in forward

Results:
- **55ms/step** (fixed speed regression, down from 78ms, still 9ms overhead vs 46ms baseline)
- 3235 steps in 3 min (vs 2299 steps with conv1d, vs ~2500 baseline)
- 3-min: **1.2624** (tied with baseline 1.2619 +/- 0.0005)
- 10-min: **1.2276** (pre-quant: 1.2207), baseline: 1.2289
- Step avg: 55.3ms, 10,828 steps

## Results Summary
| Variant | Step speed | 3-min val_bpb | 10-min val_bpb |
|---------|-----------|--------------|----------------|
| Baseline | 46ms | 1.2619 | 1.2289 |
| v1 (nn.Conv1d, k=4, random) | 75-78ms | 1.2840 | — |
| v1 (nn.Conv1d, k=4, zero) | 74ms | 1.2802-1.2819 | — |
| v2 (nn.Conv1d, k=8, zero) | ~78ms | — | 1.2393-1.2399 |
| **v3 (matmul, k=4, LR=0.005)** | **55ms** | **1.2624** | **1.2276** |

**Key learning**: `nn.Conv1d` with `groups=dim` is toxic to `torch.compile`. The transpose-conv-slice-transpose pattern causes ~2x slowdown. Must use manual shifted-view matmul for compile-friendly depthwise convolution.

**Verdict**: v3 achieves a marginal improvement (-0.0013 BPB), but at the cost of 9ms/step overhead and added complexity.
