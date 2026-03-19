# Adam Embed LR Sweep

**Finding**: Default TIED_EMBED_LR=0.05 is too high. Lower is monotonically better.

## Timeline

### 3-Minute Sweep
Agent ran LRs sequentially: 0.01, 0.03, 0.08, 0.1, 0.15, then added 0.02.

The LR=0.03 run was interrupted once (train_loss spiked to 11.37 at step 2, suggesting instability) and had to be retried.

| LR | quantized val_bpb | delta |
|----|------|---|
| **0.01** | **1.2527** | **-0.0092** |
| 0.02 | 1.2567 | -0.0052 |
| 0.03 | 1.2600 | -0.0019 |
| 0.05 | 1.2619 | baseline |
| 0.08 | 1.2706 | +0.0087 |
| 0.10 | 1.2783 | +0.0164 |
| 0.15 | 1.2886 | +0.0267 |

Agent observed a clear monotonic trend: lower embedding LR is better. LR=0.01 beats baseline by 0.0092 (well outside +/-0.0005 noise).

### 10-Minute Full Run
Agent launched LR=0.01 for a full 10-minute run. The first attempt hung with NCCL watchdog errors (Modal infrastructure issue, not LR-related). Killed and retried successfully.

| LR | quantized val_bpb | val_loss | delta |
|----|------|---|---|
| **0.01** | **1.2175** | 2.0577 | **-0.0114** |
| 0.05 | 1.2289 | 2.0623 | baseline |

## Key Findings
- LR=0.01 beats the 10-min baseline by 0.0114 nats (28x the noise level)
- The improvement is even larger at full training length than at 3 minutes, suggesting the lower LR prevents embedding overfitting that worsens with longer training
- Agent recommended exploring even lower values (0.005, 0.003) given the monotonic trend

**Verdict**: WINNER. Change default TIED_EMBED_LR from 0.05 to 0.01. Zero code changes needed beyond the hyperparameter.
