# Port-complete Qiskit Metal GDS datasets

The TransmonCross and legacy CapNInterdigitalTee sweeps can be regenerated as
two-terminal GDS datasets with the same ordered port-layer convention as
GeneralizedCapNInterdigital. Geometry comes directly from each Qiskit Metal
component's qgeometry tables; the exporter does not redraw the component from
its option values.

## Layer contract

| Layer/datatype | TransmonCross | CapNInterdigitalTee |
|---|---|---|
| `1/10` | all positive poly/path qgeometry | all positive poly/path qgeometry |
| `1/11` | absent | absent |
| `1/0` | centered dynamic ground with one etch hole | centered dynamic ground with one etch hole |
| `2/0` | cross terminal at `rect_jj` | prime/top conductor at `prime_start` |
| `3/0` | claw terminal at `readout` | second/bottom conductor at `second_end` |

Each marker begins at its Qiskit Metal pin or junction boundary and extends
outward by that terminal's native ground clearance. Its transverse width is the
native pin/junction width, so the marker exactly bridges conductor to ground.
CapN's
`prime_start` and `prime_end` belong to one electrical conductor, so only one is
marked; this deliberately preserves the two-terminal capacitance interpretation.

The ground is a square centered on the conductor bounding box with side length
`5.8 * max(conductor width, conductor height)`. Qiskit Metal subtractive
qgeometry is written as the ground polygon's single hole rather than as a
filled layer. Native etch islands are connected only by grid-scale shortest
bridges, preserving their shapes while satisfying the one-hole convention.

The direct qgeometry export is important for CapN. The stock Qiskit Metal GDS
renderer does not retain every positive CPW path on datatype 10 under the
published SQuADDS role mapping. Regeneration corrects that legacy artifact and
replaces the fixed 9 mm by 6 mm simulation box with the common dynamic ground.

## Legacy v0 compatibility

Published `static-shape-v0` intentionally remains frozen: its historical
family-specific lookup ignores TransmonCross etch and the new Transmon/CapN
ports. Use the explicit `layer-semantics-v1.3` profile to build the separately
named `static-shape-v0-port-complete` variant:

```python
from squadds.layouts import PORT_COMPLETE_ROLE_PROFILE, build_static_embeddings

embeddings, schema = build_static_embeddings(
    manifest,
    design_options_by_id,
    artifact_resolver,
    role_profile=PORT_COMPLETE_ROLE_PROFILE,
)
assert schema["model"] == "static-shape-v0-port-complete"
```

This profile consumes TransmonCross `(2,0)` / `(3,0)` as ports. The default
remains the published v0 behavior, so existing embeddings
and comparisons do not silently change. Universal v1 can use
`functional_layer_roles(component_name)` through its existing `layer_roles`
argument; universal v2 already reads the versioned roles directly.

## Generate

```bash
uv run --frozen --extra gds python scripts/generate_simulation_layout_gds.py \
  transmon-cross /path/to/transmon-gds --workers 8

uv run --frozen --extra gds python scripts/generate_capn_interdigital_tee_gds.py \
  /path/to/capn-gds
```

Both generators are resumable. Existing files are skipped unless
`--overwrite` is supplied.

## Validate

Representative validation rebuilds nine evenly spaced source rows, checks the
GDS conductor and normalized ground against Qiskit Metal-derived geometry with
a symmetric-difference test, checks the exact four-layer set, one-hole ground,
ground sizing, both conductor/ground contacts, and unique conductor assignment,
and confirms that the universal v2 encoder discovers exactly two terminals:

```bash
uv run --frozen --extra gds python scripts/validate_ported_layout_gds.py \
  transmon-cross /path/to/transmon-gds --report transmon-validation.json

uv run --frozen --extra gds python scripts/validate_ported_layout_gds.py \
  capn-interdigital-tee /path/to/capn-gds --report capn-validation.json
```

Use `--all --skip-v2` for a fast structural pass over every row, then run the
default representative v2 checks. A report is successful only if the round-trip
qgeometry, ordered layers, marker fidelity, two connected signal components,
unique marker-to-component assignment, and marker contact checks all pass.

After validation, stage the generated files with
`stage_simulation_layout_dataset.py` (TransmonCross) or
`build_capn_interdigital_tee_layout_dataset.py` (CapN). The staging commands now
write `metadata/layer-semantics-v1.json` schema 1.3.0 with the unified roles.
Pass the Transmon manifest to CapN's `--existing-manifest` option when building
a combined two-family release; the manifest and geometry-feature table are then
merged instead of overwritten.
