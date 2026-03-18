---
name: experiment
description: Run an architecture/training experiment end-to-end
user_invocable: true
args: description of the experiment to run
---

# Experiment Workflow

You are given an experiment description. Follow this process:

## 1. Implement

- Modify `train_gpt.py` to implement the described change
- Keep changes minimal and focused — don't refactor unrelated code

## 2. Verify constraints

- **16MB cap**: Code + compressed model must be under 16,000,000 bytes
- **Parameter count**: Check that the model still fits. If parameters increase beyond budget, reduce `NUM_LAYERS` by 1
  to compensate
- **No network calls** during eval

## 3. Short test run (1 minute)

Run a 1-minute wallclock test to verify the idea works and check loss direction. Additionally, if there are multiple
components, test them sequentially with one minute runs and then 1 final 10 minute run:

```bash
RUN_ID=<descriptive_name> MAX_WALLCLOCK_SECONDS=60 modal run modal_train.py
```

The RUN_ID must describe the experiment (e.g., `swiglu_9L_512d`, `rope_base50k_9L_512d`). Never use generic names.

Check:

- Does it train without errors?
- Is train_loss trending in the right direction compared to baseline?
- Are there any NaN/inf issues?
- Does the quantized model size stay under 16MB?

## 4. Full run (10 minutes)

Once the short run passes, do the full evaluation:

```bash
RUN_ID=<descriptive_name>_full MAX_WALLCLOCK_SECONDS=600 modal run modal_train.py
```

Compare final `val_bpb` against the baseline (1.2244) on wandb.

## Notes

- The ~164s torch.compile startup is unavoidable — a 1-minute wallclock run will only get ~1 real training step after
  warmup, but that's enough to check for crashes and loss sanity
- Always compare against wandb baseline runs in project `parameter-golf`
- The experiment should be logged to wandb so results are tracked
- You're working in a worktree, often other experiments are running in parallel, sometimes modal capacity can overflow,
  in which case wait a bit then resume
