# SSL Embedding Pre-training

100-step SSL stage before warmup using gated causal conv (both up and gate are Conv1d, k=6, hidden=2048) to predict next-token input embedding via cosine similarity loss.

## Timeline

### Attempt 1: Tied embeddings, default LR
- SSL loss: 1.0 → 0.72
- Initial val_loss: 32 (!!!) vs normal ~6.7
- SSL destroys the tied embedding/output head
- 3-min: 1.3148 (catastrophic)

### Attempt 2: Tied + norm preservation
- Clamp embedding row norms after each SSL step
- 3-min: 1.2701, 10-min: 1.2337
- Better but still worse than baseline (1.2289)

### Attempt 3: Untied 8L, LR=1.8 (embed_lr=0.6 × mult=3.0)
- SSL loss diverged (0.57 → 0.70)
- 3-min: 1.2770 (worse)

### Attempt 4: Untied 8L, LR=0.03 (mult=0.05)
- SSL loss converged (1.0 → 0.71)
- 3-min: 1.2609 (marginally better than 1.2619 baseline)
- 10-min: 1.2299 (noise, same as untied 8L without SSL: 1.2285)

**Verdict**: SSL with tied embeddings is fundamentally broken (corrupts output head). With untied embeddings + dropped layer, any SSL benefit is eaten by the layer loss. Needs a way to untie without losing a layer.
