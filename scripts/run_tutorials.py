"""Execute tutorial notebooks top to bottom and fail on any cell error (the CI tutorials job).

    python scripts/run_tutorials.py tutorials/Tutorial-11_*.ipynb [--timeout 1200] [--out DIR]

Each notebook runs in its own folder (relative data paths work) on a fresh python3 kernel. Cells tagged ``skip-ci`` are
not run (cells that need maintainer data or hours of training); the runner prints which ones it skipped. Shell lines
(``!squadds-ml ...``) must exit with 0: the runner checks ``_exit_code`` after every cell that has one.
Prints the wall time of every notebook and its slowest cells, and the peak memory of the kernel where the OS reports it.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

SKIP_TAG = "skip-ci"


def _peak_child_mib() -> float | None:
    try:
        import resource
    except ImportError:  # Windows
        return None
    peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return peak / 2**20 if sys.platform == "darwin" else peak / 2**10


def run(path: Path, timeout: int, out: Path | None) -> tuple[bool, float]:
    import nbformat
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError

    nb = nbformat.read(path, as_version=4)
    skipped = []
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code" and SKIP_TAG in cell.metadata.get("tags", []):
            skipped.append(i)
            cell.source = f"print('skipped in CI (tag {SKIP_TAG}): cell {i}')"
        elif cell.cell_type == "code" and any(line.lstrip().startswith("!") for line in cell.source.splitlines()):
            cell.source += f"\nassert _exit_code == 0, 'shell command in cell {i} exited with ' + str(_exit_code)"
    times = {}
    client = NotebookClient(
        nb, timeout=timeout, kernel_name="python3", resources={"metadata": {"path": str(path.parent)}}
    )

    starts = {}

    def on_cell_executed(cell, cell_index, **_):
        times[cell_index] = time.perf_counter() - starts[cell_index]

    client.on_cell_start = lambda cell, cell_index, **_: starts.__setitem__(cell_index, time.perf_counter())
    client.on_cell_executed = on_cell_executed

    t0 = time.perf_counter()
    ok = True
    try:
        client.execute()
    except CellExecutionError as e:
        ok = False
        print(f"FAILED {path.name}\n{e}", flush=True)
    wall = time.perf_counter() - t0
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        nbformat.write(nb, out / path.name)

    slow = sorted(times.items(), key=lambda kv: -kv[1])[:5]
    peak = _peak_child_mib()
    print(
        f"{'OK' if ok else 'FAIL'} {path.name}: {wall:.1f} s"
        + (f", peak kernel memory {peak:.0f} MiB" if peak is not None else "")
        + (f", skipped cells {skipped}" if skipped else ""),
        flush=True,
    )
    print("  slowest cells: " + ", ".join(f"[{i}] {s:.1f} s" for i, s in slow), flush=True)
    return ok, wall


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("notebooks", nargs="+", type=Path)
    ap.add_argument("--timeout", type=int, default=1200, help="per-cell timeout in seconds")
    ap.add_argument("--out", type=Path, default=None, help="folder for the executed notebooks")
    a = ap.parse_args(argv)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("QISKIT_METAL_HEADLESS", "1")
    # console scripts (squadds-ml) of this environment, as in an activated venv
    os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")
    results = [run(p.resolve(), a.timeout, a.out) for p in a.notebooks]
    return 0 if all(ok for ok, _ in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
