# Contributing planar capacitance data

This folder is the template of a contribution to the `SQuADDS/SQuADDS_PlanarCapacitance` dataset: layouts (GDS) and their
full Maxwell capacitance matrices from a field solver. Copy it, fill it, validate it, open a pull request.

## Layout

```
manifest.yaml                              who, licence, solver, stack, setup, layer map, units, generator, family
designs.csv                                one row per design (see the header comment)
gds/<design>.gds                           the solved file, as solved (not re-saved)
sims/<design>/<sim_tag>/cap_matrix.csv     the Q3D export as is (one row per adaptive pass)
                                           or matrix.csv / matrix.parquet: long format net_a, net_b, C_fF[, pass]
sims/<design>/<sim_tag>/convergence.txt    the Q3D .conv file (optional, recommended)
sims/<design>/<sim_tag>/run.json           solver version, cores, solve_s, wall_s, peak_mem_GB, machine (optional)
nets/<design>.json                         {"C1": [0, 1], "C2": [2]}: polygon indices per net (optional)
```

`sim_tag` names one solve of a design (`q3d` for the reference solve; any other tag for a study, e.g. another stack).

## Conventions

- Capacitance: the full N x N Maxwell matrix over every conductor of the layout, ground included (N is read from the GDS;
  each conductor is its own net, never merged). Diagonal > 0, off-diagonal < 0. Any unit, declared in `manifest.yaml`.
- Ground net `G` first in `net_order`; the other conductors follow in GDS file order (the order of the first polygon of each
  conductor body on the conductor layers), unless `nets/<design>.json` names the polygons of each net.
- GDS: one top cell, unit 1 um or 1 nm, ground and conductor layers as declared in `layer_map` (default ground 1/0,
  conductors 1/10; other layers, e.g. port markers, are ignored).
- Licence: CC-BY-4.0 or CC0-1.0.

## Validate and submit

```python
from squadds.ml.capacitance.contribute import validate_capacitance_contribution, contribute_capacitance

report = validate_capacitance_contribution("my_contribution/")
print(report.to_markdown())          # errors block the submission; warnings are recorded
contribute_capacitance("my_contribution/", dry_run=True)   # the files the pull request would add
contribute_capacitance("my_contribution/")                 # opens the pull request on Hugging Face
```

The validator rejects: a missing manifest field, an unreadable GDS or invalid polygons, a conductor count that differs from
the matrix size, missing or repeated net names, asymmetry above 1e-6 (relative), a non-positive diagonal, a positive or
unexplained zero off-diagonal entry, a signal block that is not positive definite, a licence other than CC-BY-4.0 / CC0-1.0,
and a design identical (`geom_hash`, 1 nm) to a held-out test design. It warns on closure above 5 %, missing or failed
convergence, an unusual stack and duplicates of other dataset designs. Absolute paths are stripped and e-mails dropped
unless `email_public: true`. `check_prior=True` (needs `pip install "squadds[ml]"`) also compares every entry with the
`optimized_physics_prior` of the layout and reports entries off by more than a factor 2.

After review the maintainers merge the pull request, compute the physics priors and embeddings on their cluster, and add
the rows with `source = "community"` and `split = "none"`.
