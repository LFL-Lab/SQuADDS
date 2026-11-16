"""Ansys Q3D capacitance solve of a layout GDS with the settings of the planar capacitance dataset (the two-transmon example).

    python run_q3d.py q3d_setup.json --cores 16

``q3d_setup.json`` (written by the two-transmon example) holds the GDS file name, the layer stack, one entry per GDS polygon of layer 1
(in file order) with its net, the per-net initial mesh seeds and the Q3D setup. Needs Ansys Electronics Desktop with Q3D and
PyAEDT. Writes ``cap_matrix.csv`` (Maxwell matrix of every adaptive pass, pF), the convergence table and ``result.json``
(final matrix in fF, passes, wall times) next to the setup file. Under Slurm, AEDT takes its core count from the task count:
request ``--ntasks=N --cpus-per-task=1`` and pass ``--cores N``; check "Cores N" in the solver profile
(``*.aedtresults/*.profile``).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import time
from pathlib import Path

import gdstk


def polygon_boxes(gds: Path, layer: int = 1) -> list[tuple]:
    cell = gdstk.read_gds(str(gds), 1e-3).top_level()[0]
    return [tuple(map(float, (*p.bounding_box()[0], *p.bounding_box()[1]))) for p in cell.polygons if p.layer == layer]


def name_objects(q3d, gds: Path, polygons: list[dict]) -> None:
    """Rename each imported object to its polygon's name by exact bounding-box matching (mm)."""
    boxes = polygon_boxes(gds)
    objects = list(q3d.modeler.object_names)
    if len(boxes) != len(polygons) or len(objects) != len(polygons):
        raise RuntimeError(
            f"GDS has {len(boxes)} layer-1 polygons, imported {len(objects)}, setup lists {len(polygons)}"
        )
    mapping = {}
    for obj in objects:
        bb = q3d.modeler[obj].bounding_box
        d = [sum(abs(x - y) for x, y in zip((bb[0], bb[1], bb[3], bb[4]), b)) for b in boxes]
        k = min(range(len(d)), key=d.__getitem__)
        if d[k] > 1e-3 or k in mapping.values():
            raise RuntimeError(f"no unique bounding-box match for {obj} (best {polygons[k]['name']}, {d[k]:.2e} mm)")
        mapping[obj] = k
    for obj, k in mapping.items():
        q3d.modeler[obj].name = polygons[k]["name"]


def final_matrix(path: Path, nets: list[str]) -> dict:
    """Last solved pass of the per-pass report (rows after it are NaN placeholders), in fF, with symmetry and closure."""
    rows = list(csv.reader(open(path)))
    hdr, data = rows[0], [r for r in rows[1:] if r and not any(x.strip().lower() == "nan" for x in r[1:])]
    if not data:
        raise RuntimeError(f"no solved pass in {path}")
    col = {h.strip(): j for j, h in enumerate(hdr)}
    C = [[1e3 * float(data[-1][col[f"C({a},{b}) [pF]"]]) for b in nets] for a in nets]
    n = len(nets)
    sym = max(abs(C[i][j] - C[j][i]) / max(abs(C[i][j]), 1e-12) for i in range(n) for j in range(n))
    closure = {nets[i]: sum(C[i]) / C[i][i] for i in range(n)}
    ok = all(math.isfinite(v) for r in C for v in r) and sym < 1e-3
    return {"nets": nets, "C_fF": C, "passes": len(data), "symmetry_rel": sym, "closure": closure, "qc_ok": ok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("setup")
    ap.add_argument("--cores", type=int, default=16)
    ap.add_argument("--version", default="2025.2")
    a = ap.parse_args()
    from ansys.aedt.core import Desktop
    from ansys.aedt.core.q3d import Q3d

    t_start = time.time()
    setup_file = Path(a.setup).resolve()
    cfg = json.loads(setup_file.read_text())
    gds = setup_file.parent / cfg["gds"]
    polygons = cfg["polygons"]
    aedt = setup_file.with_suffix(".aedt")
    for stale in (aedt, aedt.with_suffix(".aedt.lock")):
        stale.unlink(missing_ok=True)
    desk = Desktop(version=a.version, non_graphical=True, new_desktop=True, close_on_exit=True)
    q3d = Q3d(project=str(aedt), design="q3d", version=a.version, new_desktop=False, remove_lock=True)
    q3d.materials.duplicate_material("silicon", "cryo_silicon")
    q3d.materials["cryo_silicon"].permittivity = cfg["layer_stack"]["permittivity"]
    q3d.materials["cryo_silicon"].dielectric_loss_tangent = 1e-7
    q3d.import_gds_3d(input_file=str(gds), mapping_layers={1: (0, 0)})
    name_objects(q3d, gds, polygons)
    bb = q3d.modeler.get_model_bounding_box()
    q3d.modeler.create_box(
        origin=[bb[0], bb[1], bb[2]],
        sizes=[bb[3] - bb[0], bb[4] - bb[1], -cfg["layer_stack"]["substrate_thickness_um"] * 1e-3],
        name="substrate",
        material="cryo_silicon",
    )
    nets = {}
    for p in polygons:
        nets.setdefault(p["net"], []).append(p["name"])
    for net, objects in nets.items():
        if len(objects) > 1:  # overlapping sheets of one body (e.g. cross + IDC lead) must be one object for Q3D
            q3d.modeler.unite(objects)
            nets[net] = [objects[0]]
    q3d.assign_thin_conductor([o for objects in nets.values() for o in objects], name="thin_cond")
    for net, objects in nets.items():
        q3d.assign_net(assignment=objects, net_name=net, net_type="Signal")
        seed = cfg["mesh_seed_um"].get(net)
        if seed:
            q3d.mesh.assign_length_mesh(
                assignment=objects,
                inside_selection=False,
                maximum_length=f"{seed}um",
                maximum_elements=None,
                name=f"seed_{net}",
            )
    setup = q3d.create_setup("setup", **cfg["q3d_setup"])
    setup.props.pop("DC", None)
    setup.props.pop("AC", None)
    setup.update()
    q3d.save_project()
    t0 = time.time()
    if not q3d.analyze_setup("setup", cores=a.cores):
        msgs = [str(m) for m in q3d.odesktop.GetMessages(q3d.project_name, q3d.design_name, 0)]
        raise RuntimeError("Q3D analyze failed: " + " | ".join(msgs[-8:]))
    solve_s = time.time() - t0
    order = list(nets)
    traces = [f"C({x},{y})" for x in order for y in order]
    rep = q3d.odesign.GetModule("ReportSetup")
    rep.CreateReport(
        "cap_matrix",
        "Matrix",
        "Data Table",
        "setup : AdaptivePass",
        ["Context:=", "Original"],
        ["Pass:=", ["All"], "Freq:=", ["All"]],
        ["X Component:=", "Pass", "Y Component:=", traces],
    )
    out = setup_file.parent / "cap_matrix.csv"
    rep.ExportToFile("cap_matrix", str(out), False)
    q3d.odesign.ExportConvergence("setup", "", "CG", str(aedt.with_suffix(".conv")))
    res = final_matrix(out, order)
    res.update(solve_s=solve_s, wall_s=time.time() - t_start, cores=a.cores, host=platform.node(), version=a.version)
    (setup_file.parent / "result.json").write_text(json.dumps(res, indent=1))
    print(
        f"solved in {solve_s:.0f} s on {a.cores} cores, {res['passes']} passes; QC {'ok' if res['qc_ok'] else 'FAILED'}"
    )
    for x, row in zip(order, res["C_fF"]):
        print(f"{x:>10}", " ".join(f"{v:9.3f}" for v in row))
    q3d.save_project()
    desk.release_desktop(close_projects=True, close_on_exit=True)


if __name__ == "__main__":
    main()
