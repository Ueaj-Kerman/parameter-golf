# Canon Layers (Depthwise Causal Conv1d)

From "Physics of Language Models: Part 4.1" (Allen-Zhu). Depthwise causal conv1d per block with skip connection and zero-init.

## Iterations
### v1: Pre-activation, nn.Conv1d, k=4
- 75-78ms/step (2x baseline 46ms)
- transpose-conv-transpose pattern breaks torch.compile fusion
- 3-min: 1.2840 (terrible)
- 10-min: 1.2393

### v2: Post-activation, nn.Conv1d, k=8, zero-init
- Still 78ms/step
- 10-min: 1.2399

### v3: Post-activation, pure matmul (shifted views), k=4, CANON_LR=0.005
- 55ms/step (fixed speed regression, still 9ms overhead)
- 3-min: 1.2624 (tied with baseline)
- 10-min: 1.2276 (-0.0013 vs baseline)

**Key learning**: nn.Conv1d with groups=dim is toxic to torch.compile. Must use manual shifted-view matmul.

**Verdict**: Marginal improvement (-0.0013), not worth the complexity and 9ms/step overhead.
