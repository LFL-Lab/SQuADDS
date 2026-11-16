"""``squadds-ml predict``: GDS -> full Maxwell capacitance matrix (fF, ground first) from the command line."""

from __future__ import annotations

import argparse
import atexit
import json
import sys


def _layers(s: str) -> list[tuple[int, int]]:
    return [tuple(int(v) for v in x.split("/")) for x in s.split(",") if x]


def _no_exit_ansi_reset() -> None:
    """numba colours its messages with colorama, whose exit hook writes an ANSI reset to a terminal stdout: drop that hook
    so CSV / JSON output ends with the data."""
    ini = sys.modules.get("colorama.initialise")
    if ini is not None:
        atexit.unregister(ini.reset_all)


def cmd_predict(a) -> int:
    from squadds.ml import require_ml

    require_ml()
    from .models import load_model
    from .predictor import CapacitancePredictor

    model = load_model(a.model, path=a.model_path, revision=a.revision)
    pred = CapacitancePredictor(model, threads=a.threads)
    kw = {"ground_layers": _layers(a.ground), "conductor_layers": _layers(a.conductor)}
    if a.ground_pieces:
        kw["ground_pieces"] = [int(x) for x in a.ground_pieces.split(",") if x.strip()]
    res = pred.predict(
        a.gds if len(a.gds) > 1 else a.gds[0],
        eps_r=a.eps_r,
        thickness_um=a.thickness_um,
        workers=a.workers,
        **kw,
    )
    res = res if isinstance(res, list) else [res]
    _no_exit_ansi_reset()
    if a.format == "json":
        out = [r.to_dict() for r in res]
        print(json.dumps(out if len(out) > 1 else out[0], indent=1))
    else:
        for path, r in zip(a.gds, res):
            print(f"# {path}: {r.model}, {r.timing_s['total_s']:.3g} s ({r.threads} threads, {r.machine})")
            for w in r.warnings:
                print(f"# warning: {w}")
            print(r.to_dataframe().to_csv(float_format="%.6g"), end="")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="squadds-ml", description="SQuADDS planar capacitance models")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict", help="GDS -> full N x N Maxwell capacitance matrix (fF, ground first)")
    p.add_argument("gds", nargs="+")
    p.add_argument("--model", default="QCap", help="QCap (default) or QCap-Fast")
    p.add_argument(
        "--model-path", default=None, help="local model bundle (else $SQUADDS_CAPML_MODELS, else Hugging Face)"
    )
    p.add_argument("--revision", default=None)
    p.add_argument(
        "--threads",
        type=int,
        default=None,
        help="CPU threads (default: all usable cores, at most one per 2.5 GB of RAM and 32)",
    )
    p.add_argument("--workers", type=int, default=1, help="processes for several GDS files")
    p.add_argument("--ground", default="1/0", help="ground layers, e.g. 1/0,2/0")
    p.add_argument("--conductor", default="1/10", help="conductor layers, e.g. 1/10")
    p.add_argument(
        "--ground-pieces",
        default=None,
        help="ground and conductors on one layer: the metal pieces that form the ground (numbers as in the figure of "
        "inspect_layout, e.g. 1 or 1,4); default = detected automatically, with a warning",
    )
    p.add_argument("--eps-r", type=float, default=11.45, help="substrate permittivity (the prior is solved at it)")
    p.add_argument(
        "--thickness-um", type=float, default=350.0, help="substrate thickness in µm (the prior is solved at it)"
    )
    p.add_argument("--format", default="csv", choices=["csv", "json"])
    p.set_defaults(func=cmd_predict)
    from .train import add_train_subcommand

    add_train_subcommand(sub)
    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
