"""A generic SLURM array script for retraining (one seed per array task). No site-specific accounts, partitions or tooling:
pass your own, or leave them out and let your cluster's defaults apply."""

from __future__ import annotations

import shlex
import sys
from pathlib import Path


def write_slurm_array(
    out_path,
    shard_dir,
    seeds=(0, 1, 2),
    out_dir=None,
    partition: str | None = None,
    account: str | None = None,
    cpus: int = 2,
    time: str = "01:00:00",
    mem: str = "6G",
    python: str = sys.executable,
    arm: str = "full_gnn",
    steps: int = 18000,
    device: str = "cpu",
    gpus: int = 0,
    job_name: str = "squadds-ml-train",
    extra_sbatch: tuple = (),
    setup: str = "",
) -> Path:
    """Write ``out_path``: ``sbatch out_path`` trains ``seeds`` (array task k -> seeds[k]) on ``shard_dir`` into
    ``out_dir/s<seed>`` with ``cpus`` threads each. ``setup``: shell lines run before training (e.g. ``module load ...``,
    ``source venv/bin/activate``). Returns the path."""
    seeds = [int(s) for s in seeds]
    if not seeds:
        raise ValueError("no seeds")
    shard_dir = Path(shard_dir).resolve()
    out_dir = Path(out_dir).resolve() if out_dir else shard_dir.parent / "runs"
    out_path = Path(out_path)
    log_dir = out_dir / "logs"
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --array=0-{len(seeds) - 1}",
        "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={int(cpus)}",
        f"#SBATCH --mem={mem}",
        f"#SBATCH --time={time}",
        f"#SBATCH --output={log_dir}/%x_%A_%a.out",
    ]
    if partition:
        lines.append(f"#SBATCH --partition={partition}")
    if account:
        lines.append(f"#SBATCH --account={account}")
    if gpus:
        lines.append(f"#SBATCH --gpus={int(gpus)}")
    lines += [f"#SBATCH {x}" for x in extra_sbatch]
    lines += [
        "set -euo pipefail",
        *([setup] if setup else []),
        f"export OMP_NUM_THREADS={int(cpus)} MKL_NUM_THREADS={int(cpus)} OPENBLAS_NUM_THREADS={int(cpus)}",
        f"SEEDS=({' '.join(map(str, seeds))})",
        "SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}",
        f"mkdir -p {shlex.quote(str(out_dir))}",
        " ".join(
            [
                shlex.quote(python),
                "-m squadds.ml.capacitance.train",
                f"--data {shlex.quote(str(shard_dir))}",
                f"--out {shlex.quote(str(out_dir))}",
                f"--arm {arm}",
                '--seeds "$SEED"',
                f"--steps {int(steps)}",
                f"--threads {int(cpus)}",
                f"--device {device}",
            ]
        ),
        "",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    out_path.chmod(0o755)
    return out_path
