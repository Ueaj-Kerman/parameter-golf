# Untied Embeddings with Zero-Init Head

Separate input embedding and output projection. Output head zero-initialized (already implemented in codebase via `_zero_init = True` on `lm_head`).

## Timeline

### Implementation
Agent found the change was minimal — `lm_head` zero-init was already implemented in existing code. When `tie_embeddings=False`, `lm_head` is created as `CastedLinear` with `_zero_init = True`, and `_init_weights()` calls `nn.init.zeros_()`.

LR config already existed: `embed_lr=0.6` (input embedding), `head_lr=0.008` (output head). Only change needed: set `TIE_EMBEDDINGS` default from `"1"` to `"0"`.

Agent estimated parameter budget: with 1024 vocab and 512 dim, untying adds 524,288 params. Expected to fit under 16MB.

### 1-Minute Smoke Test (9L untied)
- No errors or NaN, loss: 6.93 -> 2.22 by step 1299
- val_bpb: 1.3148 (quantized: 1.3157 — minimal quant degradation)
- Model params: 17,584,200
- Compressed size: 13,712,608 bytes — well under 16MB cap

### 10-Minute Run (9L untied)
- **val_bpb: 1.2153 (pre-quant), 1.2168 (post-quant)** — improvement of 0.0076 vs baseline
- 13,012 steps in 600s
- **Compressed size: 16,307,852 bytes — exceeds 16MB cap by ~308KB**
- Agent immediately identified the size issue and decided to reduce to 8 layers

### 8L Smoke Test
- Compressed size: 12,708,483 bytes — well under cap
- 15,747,136 params
- Faster per-step (~41ms vs ~46ms) due to fewer layers

### 10-Minute Run (8L untied) — First Run
- val_bpb: 1.2266 (pre-quant), 1.2285 (post-quant)
- val_loss: 2.0711 (pre-quant), 2.0743 (post-quant)
- Compressed size: 14,602,090 bytes
- 14,659 steps / 20,000 in 600s

### 10-Minute Run (8L untied) — Second Run (rerun agent)
Second agent verified config was already set (NUM_LAYERS=8, TIE_EMBEDDINGS=0). Ran 3-min and 10-min tests.
- 3-min: 1.2602 quantized (baseline: 1.2614) — slightly better
- 10-min: **1.2284** quantized (baseline: 1.2287) — essentially tied
- Compressed size: 14,602,068 bytes
- Step speed: 40.9ms vs 47.9ms baseline (faster due to fewer layers), gets 14,659 vs 12,527 steps

## Results
| Config | 10-min quantized val_bpb | val_loss | Size | delta |
|--------|------|---|---|---|
| 9L untied | 1.2168 | 2.0522 | **16.3MB (OVER CAP)** | -0.0121 |
| 8L untied | 1.2284-1.2285 | 2.0711-2.1113 | 14.6MB | -0.0004 |
| 9L tied (baseline) | 1.2289 | 2.0623 | 15.9MB | — |

## Key Findings
- 9L untied shows real potential (-0.012 BPB) but exceeds 16MB cap by 308KB
- 8L untied fits (14.6MB) but the lost layer capacity cancels the untying gain
- Faster per-step (40.9ms vs 47.9ms) due to fewer layers, gets ~17% more steps
- Two independent runs of 8L untied give consistent results (1.2284 and 1.2285)
- Agent noted `head_lr=0.008` may need further tuning to close the gap

**Verdict**: Net neutral at 8L. Need compression improvements to fit 9L untied under cap.
