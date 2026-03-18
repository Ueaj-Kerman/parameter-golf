"""
Modal app for Parameter Golf: train on 8xH100 with cached image and data.

Usage:
    # Download data to volume (run once)
    modal run modal_train.py::download_data

    # Launch training (give every run a descriptive RUN_ID!)
    RUN_ID=baseline_9L_512d modal run modal_train.py

    # Launch with wandb disabled (for clean timing benchmarks)
    RUN_ID=baseline_9L_512d WANDB=0 modal run modal_train.py

    # Launch with custom hyperparams
    RUN_ID=wider_12L_768d NUM_LAYERS=12 MODEL_DIM=768 modal run modal_train.py

    # Retrieve artifacts
    modal volume get parameter-golf-data outputs/final_model.int8.ptz .
    modal volume get parameter-golf-data outputs/ ./outputs
"""

from __future__ import annotations

import os
import subprocess

import modal

# ---------------------------------------------------------------------------
# Image: all packages baked in so cold starts only pull the cached layers.
# Uses CUDA devel image for torch.compile / custom kernels.
# ---------------------------------------------------------------------------
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu24.04", add_python="3.12"
    )
    .apt_install("git")
    .pip_install(
        "numpy",
        "tqdm",
        "torch==2.10",
        "huggingface-hub",
        "kernels",
        "setuptools",
        "typing-extensions==4.15.0",
        "datasets",
        "tiktoken",
        "sentencepiece",
        "wandb",
    )
    .env({"NCCL_DEBUG": "WARN"})
    .add_local_dir("data", remote_path="/root/project/data")
    .add_local_file("train_gpt.py", remote_path="/root/project/train_gpt.py")
)

# ---------------------------------------------------------------------------
# Persistent volume: holds downloaded shards + training outputs.
# ---------------------------------------------------------------------------
volume = modal.Volume.from_name("parameter-golf-data", create_if_missing=True)
VOLUME_PATH = "/data"

app = modal.App(
    "parameter-golf",
    image=image,
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("wandb-secret")],
)

# ---------------------------------------------------------------------------
# Data download — run once, results persist on the volume.
# ---------------------------------------------------------------------------
@app.function(timeout=30 * 60)
def download_data(variant: str = "sp1024", train_shards: int = 80):
    """Download FineWeb shards and tokenizer to the persistent volume."""
    from pathlib import Path

    data_dir = Path(VOLUME_PATH) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    src_data = Path("/root/project/data")
    datasets_link = src_data / "datasets"
    tokenizers_link = src_data / "tokenizers"

    vol_datasets = data_dir / "datasets"
    vol_tokenizers = data_dir / "tokenizers"
    vol_datasets.mkdir(parents=True, exist_ok=True)
    vol_tokenizers.mkdir(parents=True, exist_ok=True)

    if not datasets_link.exists():
        datasets_link.symlink_to(vol_datasets)
    if not tokenizers_link.exists():
        tokenizers_link.symlink_to(vol_tokenizers)

    subprocess.run(
        [
            "python", "-u", str(src_data / "cached_challenge_fineweb.py"),
            "--variant", variant,
            "--train-shards", str(train_shards),
        ],
        check=True,
    )
    volume.commit()
    print(f"Done — {variant}, {train_shards} train shards → {vol_datasets}")


# ---------------------------------------------------------------------------
# Training — 8×H100, 10 min wallclock target, optional wandb logging.
# Set WANDB=0 to disable (for clean timing benchmarks).
# ---------------------------------------------------------------------------
TRAIN_ENV_KEYS = [
    "RUN_ID", "SEED", "ITERATIONS", "WARMDOWN_ITERS", "WARMUP_STEPS",
    "TRAIN_BATCH_TOKENS", "TRAIN_SEQ_LEN", "MAX_WALLCLOCK_SECONDS",
    "VOCAB_SIZE", "NUM_LAYERS", "NUM_KV_HEADS", "MODEL_DIM", "NUM_HEADS",
    "MLP_MULT", "TIE_EMBEDDINGS", "ROPE_BASE", "LOGIT_SOFTCAP",
    "EMBED_LR", "HEAD_LR", "TIED_EMBED_LR", "MATRIX_LR", "SCALAR_LR",
    "MUON_MOMENTUM", "VAL_LOSS_EVERY", "TRAIN_LOG_EVERY",
]


def parse_log_line(line: str) -> dict | None:
    """Parse a training log line into a dict of metrics."""
    if not line.startswith("step:"):
        return None
    metrics = {}
    for token in line.split():
        if ":" not in token:
            continue
        key, _, val = token.partition(":")
        if key == "step":
            cur, _, total = val.partition("/")
            try:
                metrics["step"] = int(cur)
                metrics["total_steps"] = int(total)
            except ValueError:
                pass
        else:
            try:
                metrics[key] = float(val.rstrip("ms"))
            except ValueError:
                pass
    return metrics if metrics else None


@app.function(gpu="H100:8", timeout=30 * 60)
def train(env_overrides: dict[str, str] | None = None):
    """Run distributed training via torchrun on 8×H100."""
    import shutil
    from pathlib import Path

    env_overrides = env_overrides or {}
    compile_only = env_overrides.pop("COMPILE_ONLY", None) == "1"
    use_wandb = not compile_only and env_overrides.get("WANDB", os.environ.get("WANDB", "1")) != "0"
    wb_run = None

    dataset_dir = Path(VOLUME_PATH) / "data" / "datasets" / "fineweb10B_sp1024"
    tokenizer_path = Path(VOLUME_PATH) / "data" / "tokenizers" / "fineweb_1024_bpe.model"
    if not dataset_dir.exists():
        raise RuntimeError(
            f"Dataset not found at {dataset_dir}. "
            "Run `modal run modal_train.py::download_data` first."
        )

    env = {
        **os.environ,
        "DATA_PATH": str(dataset_dir),
        "TOKENIZER_PATH": str(tokenizer_path),
        "MAX_WALLCLOCK_SECONDS": "60",
        "TORCHINDUCTOR_CACHE_DIR": f"{VOLUME_PATH}/torch_cache",
        **env_overrides,
    }
    if compile_only:
        env["TIMING_ONLY"] = "1"

    run_id = env.get("RUN_ID", "unnamed")
    config = {k: env.get(k) for k in TRAIN_ENV_KEYS if env.get(k) is not None}

    if use_wandb:
        import wandb
        wb_run = wandb.init(project="parameter-golf", name=run_id, config=config)

    train_script = "/root/project/train_gpt.py"
    proc = subprocess.Popen(
        [
            "torchrun",
            "--standalone",
            "--nproc_per_node", "8",
            train_script,
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    for line in proc.stdout:
        line = line.rstrip()
        print(line)

        if wb_run:
            metrics = parse_log_line(line)
            if metrics:
                step = metrics.pop("step", None)
                metrics.pop("total_steps", None)
                if step is not None:
                    wb_run.log(metrics, step=step)

            if "final_int8_zlib_roundtrip_exact" in line:
                for token in line.split():
                    if ":" in token:
                        k, _, v = token.partition(":")
                        try:
                            wb_run.summary[f"final_{k}"] = float(v)
                        except ValueError:
                            pass

    proc.wait()
    if proc.returncode != 0:
        if wb_run:
            wb_run.finish(exit_code=1)
        raise subprocess.CalledProcessError(proc.returncode, train_script)

    if compile_only:
        volume.commit()
        print("Compile-only run complete. Torch cache saved to volume.")
        return

    # Persist outputs to volume
    out = Path(VOLUME_PATH) / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    import glob as g
    for pattern in ("/root/final_model.*", "/root/logs/*.txt"):
        for f in g.glob(pattern):
            shutil.copy2(f, out / Path(f).name)

    volume.commit()
    if wb_run:
        wb_run.finish()
    print(f"Training complete. Outputs at {out}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main():
    # Forward TRAIN_ENV_KEYS + WANDB from local shell into the remote container
    env_overrides = {}
    for key in [*TRAIN_ENV_KEYS, "WANDB", "COMPILE_ONLY", "TIMING_ONLY"]:
        val = os.environ.get(key)
        if val is not None:
            env_overrides[key] = val
    train.remote(env_overrides=env_overrides if env_overrides else None)
