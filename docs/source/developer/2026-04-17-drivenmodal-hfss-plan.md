# Driven-Modal HFSS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a production-ready HFSS driven-modal API to SQuADDS for capacitance extraction and coupled-system post-processing, plus two tutorials and a Hugging Face-ready dataset contract.

**Architecture:** Keep `AnsysSimulator` as the public facade, but implement driven-modal support in a focused `squadds/simulations/drivenmodal/` package. Separate request modeling, layer-stack resolution, port generation, HFSS execution, artifact checkpointing, post-processing, and dataset serialization so the feature can be validated and extended without tangling it into the existing eigenmode/LOM helpers.

**Tech Stack:** Python 3.10+, Qiskit Metal, pyEPR/pyaedt integration through `QHFSSRenderer`, NumPy, pandas, SciPy, scqubits, scikit-rf, pytest, Sphinx/MyST, Jupyter notebooks.

---

## File Structure

### Create

- `docs/source/developer/2026-04-17-drivenmodal-hfss-worklog.md`
- `squadds/simulations/drivenmodal/__init__.py`
- `squadds/simulations/drivenmodal/models.py`
- `squadds/simulations/drivenmodal/layer_stack.py`
- `squadds/simulations/drivenmodal/ports.py`
- `squadds/simulations/drivenmodal/artifacts.py`
- `squadds/simulations/drivenmodal/hfss_runner.py`
- `squadds/simulations/drivenmodal/capacitance.py`
- `squadds/simulations/drivenmodal/coupled_postprocess.py`
- `squadds/simulations/drivenmodal/dataset_records.py`
- `tests/test_drivenmodal_models.py`
- `tests/test_drivenmodal_layer_stack.py`
- `tests/test_drivenmodal_ports.py`
- `tests/test_drivenmodal_artifacts.py`
- `tests/test_drivenmodal_capacitance.py`
- `tests/test_drivenmodal_coupled_postprocess.py`
- `tests/test_drivenmodal_dataset_records.py`
- `tests/test_drivenmodal_docs.py`
- `tests/fixtures/drivenmodal/README.md`
- `docs/source/tutorials/Tutorial-10_DrivenModal_Capacitance_Extraction.ipynb`
- `docs/source/tutorials/Tutorial-11_DrivenModal_Coupled_System_Postprocessing.ipynb`

### Modify

- `docs/source/developer/2026-04-17-drivenmodal-hfss-prd.md`
- `docs/source/developer/2026-04-17-drivenmodal-hfss-plan.md`
- `squadds/simulations/ansys_simulator.py`
- `squadds/simulations/__init__.py`
- `pyproject.toml`
- `docs/source/tutorials/index.rst`
- `README.md`
- `docs/source/release_notes.rst`
- `tests/test_ansys_simulator.py`

---

### Task 0: Establish the single-source-of-truth work log

**Files:**
- Create: `docs/source/developer/2026-04-17-drivenmodal-hfss-worklog.md`
- Modify: `docs/source/developer/2026-04-17-drivenmodal-hfss-prd.md`
- Modify: `docs/source/developer/2026-04-17-drivenmodal-hfss-plan.md`

- [ ] **Step 1: Create the work log before starting code**

```markdown
# Driven-Modal HFSS Work Log

## Canonical docs
- PRD: `docs/source/developer/2026-04-17-drivenmodal-hfss-prd.md`
- Plan: `docs/source/developer/2026-04-17-drivenmodal-hfss-plan.md`

## Branch
- `codex/drivenmodal-api-prd`

## Rules for future agents
- Do not touch unrelated untracked local tutorial/runtime files.
- Keep materials fixed in v1; only thickness-style layer-stack overrides are allowed.
- Keep checkpoint manifests resume-safe; never introduce sweep execution that cannot restart from partial progress.
- Prefer modular, reusable simulation helpers over system-specific one-off scripts.

## Current status
- [ ] Task 0 work log created
- [ ] Task 1 scaffold started
- [ ] Task 2 models started

## Next concrete step
- Write the first failing import and model tests.
```

- [ ] **Step 2: Record restart / handoff conventions**

Update the work log with sections for:

- current implementation slice
- touched files
- latest verification commands and outputs
- open decisions
- next safe restart point

- [ ] **Step 3: Re-read the PRD and plan after adding the work log**

Run: `sed -n '1,120p' docs/source/developer/2026-04-17-drivenmodal-hfss-prd.md`
Expected: The PRD references the work log as the canonical execution-status record

Run: `sed -n '1,120p' docs/source/developer/2026-04-17-drivenmodal-hfss-plan.md`
Expected: The plan includes the work log in file structure and Task 0

- [ ] **Step 4: Commit**

```bash
git add docs/source/developer/2026-04-17-drivenmodal-hfss-worklog.md docs/source/developer/2026-04-17-drivenmodal-hfss-prd.md docs/source/developer/2026-04-17-drivenmodal-hfss-plan.md
git commit -m "docs: add drivenmodal work log and handoff conventions"
```

---

### Task 1: Add the driven-modal package scaffold and dependency entry

**Files:**
- Create: `squadds/simulations/drivenmodal/__init__.py`
- Modify: `pyproject.toml`
- Modify: `squadds/simulations/__init__.py`
- Test: `tests/imports_test.py`

- [ ] **Step 1: Write the failing import test**

```python
def test_drivenmodal_package_exports():
    from squadds.simulations.drivenmodal import (
        CapacitanceExtractionRequest,
        CoupledSystemDrivenModalRequest,
        DrivenModalLayerStackSpec,
    )

    assert CapacitanceExtractionRequest is not None
    assert CoupledSystemDrivenModalRequest is not None
    assert DrivenModalLayerStackSpec is not None
```

- [ ] **Step 2: Run the focused test to verify it fails**

Run: `uv run pytest tests/imports_test.py -q --tb=short`
Expected: FAIL with an import error for `squadds.simulations.drivenmodal`

- [ ] **Step 3: Add the new dependency and export scaffold**

```toml
dependencies = [
    # ...
    "scikit-rf>=1.2",
]
```

```python
# squadds/simulations/drivenmodal/__init__.py
from .models import (
    CapacitanceExtractionRequest,
    CoupledSystemDrivenModalRequest,
    DrivenModalLayerStackSpec,
)

__all__ = [
    "CapacitanceExtractionRequest",
    "CoupledSystemDrivenModalRequest",
    "DrivenModalLayerStackSpec",
]
```

```python
# squadds/simulations/__init__.py
from .ansys_simulator import AnsysSimulator
from .drivenmodal import (
    CapacitanceExtractionRequest,
    CoupledSystemDrivenModalRequest,
    DrivenModalLayerStackSpec,
)
```

- [ ] **Step 4: Run the import test again**

Run: `uv run pytest tests/imports_test.py -q --tb=short`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml squadds/simulations/__init__.py squadds/simulations/drivenmodal/__init__.py tests/imports_test.py
git commit -m "feat: scaffold drivenmodal package and dependency surface"
```

---

### Task 2: Define typed request/result models and validation rules

**Files:**
- Create: `squadds/simulations/drivenmodal/models.py`
- Test: `tests/test_drivenmodal_models.py`

- [ ] **Step 1: Write the failing model tests**

```python
import pytest

from squadds.simulations.drivenmodal.models import (
    CapacitanceExtractionRequest,
    CoupledSystemDrivenModalRequest,
    DrivenModalLayerStackSpec,
    DrivenModalSweepSpec,
)


def test_capacitance_request_rejects_unknown_system_kind():
    with pytest.raises(ValueError, match="system_kind"):
        CapacitanceExtractionRequest(
            system_kind="bad_kind",
            design_payload={"design_options": {}},
            layer_stack=DrivenModalLayerStackSpec(),
            sweep=DrivenModalSweepSpec(start_ghz=1.0, stop_ghz=10.0, count=101),
        )


def test_coupled_request_rejects_unknown_resonator_type():
    with pytest.raises(ValueError, match="resonator_type"):
        CoupledSystemDrivenModalRequest(
            resonator_type="bad_type",
            design_payload={"design_options": {}},
            layer_stack=DrivenModalLayerStackSpec(),
            sweep=DrivenModalSweepSpec(start_ghz=4.0, stop_ghz=9.0, count=801),
        )
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `uv run pytest tests/test_drivenmodal_models.py -q --tb=short`
Expected: FAIL because `models.py` does not exist yet

- [ ] **Step 3: Implement the minimal typed models**

```python
from dataclasses import dataclass, field


@dataclass(frozen=True)
class DrivenModalLayerStackSpec:
    preset: str = "squadds_hfss_v1"
    substrate_thickness_um: float = 750.0
    metal_thickness_um: float = 0.2


@dataclass(frozen=True)
class DrivenModalSweepSpec:
    start_ghz: float
    stop_ghz: float
    count: int

    def __post_init__(self):
        if self.stop_ghz <= self.start_ghz:
            raise ValueError("stop_ghz must be greater than start_ghz")
        if self.count < 2:
            raise ValueError("count must be at least 2")


@dataclass(frozen=True)
class CapacitanceExtractionRequest:
    system_kind: str
    design_payload: dict
    layer_stack: DrivenModalLayerStackSpec
    sweep: DrivenModalSweepSpec

    def __post_init__(self):
        if self.system_kind not in {"qubit_claw", "ncap"}:
            raise ValueError("system_kind must be 'qubit_claw' or 'ncap'")


@dataclass(frozen=True)
class CoupledSystemDrivenModalRequest:
    resonator_type: str
    design_payload: dict
    layer_stack: DrivenModalLayerStackSpec
    sweep: DrivenModalSweepSpec

    def __post_init__(self):
        if self.resonator_type not in {"quarter_wave", "half_wave"}:
            raise ValueError("resonator_type must be 'quarter_wave' or 'half_wave'")
```

- [ ] **Step 4: Run the focused tests again**

Run: `uv run pytest tests/test_drivenmodal_models.py -q --tb=short`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add squadds/simulations/drivenmodal/models.py tests/test_drivenmodal_models.py
git commit -m "feat: add drivenmodal request and result models"
```

---

### Task 3: Implement layer-stack presets and override normalization

**Files:**
- Create: `squadds/simulations/drivenmodal/layer_stack.py`
- Test: `tests/test_drivenmodal_layer_stack.py`

- [ ] **Step 1: Write the failing layer-stack tests**

```python
from squadds.simulations.drivenmodal.layer_stack import resolve_layer_stack
from squadds.simulations.drivenmodal.models import DrivenModalLayerStackSpec


def test_resolve_layer_stack_uses_fixed_materials():
    rows = resolve_layer_stack(DrivenModalLayerStackSpec())
    materials = {row["material"] for row in rows}
    assert materials == {"pec", "silicon"}


def test_resolve_layer_stack_applies_thickness_overrides():
    rows = resolve_layer_stack(
        DrivenModalLayerStackSpec(substrate_thickness_um=500.0, metal_thickness_um=0.3)
    )
    assert any(row["thickness"] == "-500um" for row in rows if row["material"] == "silicon")
    assert any(row["thickness"] == "0.3um" for row in rows if row["material"] == "pec")
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `uv run pytest tests/test_drivenmodal_layer_stack.py -q --tb=short`
Expected: FAIL because `resolve_layer_stack` does not exist

- [ ] **Step 3: Implement layer-stack resolution**

```python
def resolve_layer_stack(spec):
    if spec.preset != "squadds_hfss_v1":
        raise ValueError("Unknown layer-stack preset")

    return [
        {
            "chip_name": "main",
            "layer": 1,
            "datatype": 0,
            "material": "pec",
            "thickness": f"{spec.metal_thickness_um}um",
            "z_coord": "0um",
            "fill": True,
        },
        {
            "chip_name": "main",
            "layer": 3,
            "datatype": 0,
            "material": "silicon",
            "thickness": f"-{spec.substrate_thickness_um}um",
            "z_coord": "0um",
            "fill": True,
        },
    ]
```

- [ ] **Step 4: Run the focused tests again**

Run: `uv run pytest tests/test_drivenmodal_layer_stack.py -q --tb=short`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add squadds/simulations/drivenmodal/layer_stack.py tests/test_drivenmodal_layer_stack.py
git commit -m "feat: add drivenmodal layer-stack presets"
```

---

### Task 4: Implement port-spec generation for capacitance and coupled-system requests

**Files:**
- Create: `squadds/simulations/drivenmodal/ports.py`
- Test: `tests/test_drivenmodal_ports.py`

- [ ] **Step 1: Write the failing port tests**

```python
from squadds.simulations.drivenmodal.ports import (
    build_capacitance_port_map,
    build_coupled_system_port_map,
)


def test_build_capacitance_port_map_for_ncap_has_named_nodes():
    port_map = build_capacitance_port_map("ncap", {"name": "NCapCoupler"})
    assert [entry["node"] for entry in port_map] == ["top", "bottom", "ground"]


def test_build_coupled_system_port_map_has_feedline_and_jj_ports():
    port_map = build_coupled_system_port_map({"qubit": "Q1", "feedline": "Feed"})
    assert [entry["kind"] for entry in port_map] == ["feedline_input", "feedline_output", "jj"]
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `uv run pytest tests/test_drivenmodal_ports.py -q --tb=short`
Expected: FAIL because the module does not exist

- [ ] **Step 3: Implement the minimal port-map builders**

```python
def build_capacitance_port_map(system_kind, design_payload):
    if system_kind == "ncap":
        return [
            {"node": "top", "impedance_ohms": 50.0},
            {"node": "bottom", "impedance_ohms": 50.0},
            {"node": "ground", "impedance_ohms": 50.0},
        ]
    if system_kind == "qubit_claw":
        return [
            {"node": "cross", "impedance_ohms": 50.0},
            {"node": "claw", "impedance_ohms": 50.0},
            {"node": "ground", "impedance_ohms": 50.0},
        ]
    raise ValueError(f"Unsupported system_kind: {system_kind}")


def build_coupled_system_port_map(design_payload):
    return [
        {"kind": "feedline_input", "impedance_ohms": 50.0},
        {"kind": "feedline_output", "impedance_ohms": 50.0},
        {"kind": "jj", "impedance_ohms": 50.0},
    ]
```

- [ ] **Step 4: Run the focused tests again**

Run: `uv run pytest tests/test_drivenmodal_ports.py -q --tb=short`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add squadds/simulations/drivenmodal/ports.py tests/test_drivenmodal_ports.py
git commit -m "feat: add drivenmodal port-spec builders"
```

---

### Task 5: Add checkpoint manifests and artifact policies

**Files:**
- Create: `squadds/simulations/drivenmodal/artifacts.py`
- Test: `tests/test_drivenmodal_artifacts.py`

- [ ] **Step 1: Write the failing artifact tests**

```python
from pathlib import Path

from squadds.simulations.drivenmodal.artifacts import initialize_run_manifest, mark_stage_complete


def test_initialize_run_manifest_creates_expected_paths(tmp_path: Path):
    manifest = initialize_run_manifest(tmp_path, run_id="demo-run")
    assert manifest["run_dir"].endswith("demo-run")
    assert (tmp_path / "demo-run" / "manifest.json").exists()


def test_mark_stage_complete_updates_stage_state(tmp_path: Path):
    manifest = initialize_run_manifest(tmp_path, run_id="demo-run")
    updated = mark_stage_complete(manifest, "prepared")
    assert updated["stages"]["prepared"]["status"] == "complete"
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `uv run pytest tests/test_drivenmodal_artifacts.py -q --tb=short`
Expected: FAIL because `artifacts.py` does not exist

- [ ] **Step 3: Implement the minimal manifest helpers**

```python
import json
from pathlib import Path


def initialize_run_manifest(root_dir, run_id):
    run_dir = Path(root_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "stages": {name: {"status": "pending"} for name in [
            "prepared",
            "rendered",
            "setup_created",
            "sweep_completed",
            "artifacts_exported",
            "postprocessed",
            "serialized",
        ]},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def mark_stage_complete(manifest, stage_name):
    manifest["stages"][stage_name]["status"] = "complete"
    Path(manifest["run_dir"], "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
```

- [ ] **Step 4: Run the focused tests again**

Run: `uv run pytest tests/test_drivenmodal_artifacts.py -q --tb=short`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add squadds/simulations/drivenmodal/artifacts.py tests/test_drivenmodal_artifacts.py
git commit -m "feat: add drivenmodal checkpoint manifests"
```

---

### Task 6: Implement Y-parameter to capacitance post-processing

**Files:**
- Create: `squadds/simulations/drivenmodal/capacitance.py`
- Test: `tests/test_drivenmodal_capacitance.py`

- [ ] **Step 1: Write the failing capacitance post-processing tests**

```python
import numpy as np

from squadds.simulations.drivenmodal.capacitance import capacitance_matrix_from_y


def test_capacitance_matrix_from_y_divides_imaginary_part_by_omega():
    freq_hz = 5e9
    y_matrix = 1j * 2 * np.pi * freq_hz * np.array([[10e-15, -2e-15], [-2e-15, 9e-15]])
    result = capacitance_matrix_from_y(freq_hz, y_matrix)
    assert np.allclose(result, [[10e-15, -2e-15], [-2e-15, 9e-15]])
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `uv run pytest tests/test_drivenmodal_capacitance.py -q --tb=short`
Expected: FAIL because the module does not exist

- [ ] **Step 3: Implement the minimal capacitance extraction**

```python
import numpy as np


def capacitance_matrix_from_y(freq_hz, y_matrix):
    if freq_hz <= 0:
        raise ValueError("freq_hz must be positive")
    omega = 2 * np.pi * freq_hz
    c_matrix = np.imag(y_matrix) / omega
    return 0.5 * (c_matrix + c_matrix.T)
```

- [ ] **Step 4: Run the focused tests again**

Run: `uv run pytest tests/test_drivenmodal_capacitance.py -q --tb=short`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add squadds/simulations/drivenmodal/capacitance.py tests/test_drivenmodal_capacitance.py
git commit -m "feat: add drivenmodal capacitance extraction"
```

---

### Task 7: Implement coupled-system post-processing from Touchstone data

**Files:**
- Create: `squadds/simulations/drivenmodal/coupled_postprocess.py`
- Create: `tests/test_drivenmodal_coupled_postprocess.py`
- Create: `tests/fixtures/drivenmodal/README.md`

- [ ] **Step 1: Write the failing coupled post-processing tests**

```python
import numpy as np

from squadds.simulations.drivenmodal.coupled_postprocess import calculate_g_from_chi


def test_calculate_g_from_chi_returns_positive_rate():
    value = calculate_g_from_chi(
        f_r_hz=6.0e9,
        f_q_hz=4.5e9,
        chi_hz=1.2e6,
        alpha_hz=-200e6,
    )
    assert value > 0
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `uv run pytest tests/test_drivenmodal_coupled_postprocess.py -q --tb=short`
Expected: FAIL because the module does not exist

- [ ] **Step 3: Implement the minimal analytical helper**

```python
import numpy as np


def calculate_g_from_chi(*, f_r_hz, f_q_hz, chi_hz, alpha_hz):
    omega_r = 2 * np.pi * f_r_hz
    omega_q = 2 * np.pi * f_q_hz
    chi = 2 * np.pi * chi_hz
    alpha = 2 * np.pi * alpha_hz
    delta = omega_q - omega_r
    sigma = omega_q + omega_r
    rwa = alpha / (delta * (delta - alpha))
    non_rwa = alpha / (sigma * (sigma + alpha))
    return np.sqrt(chi / (2 * (rwa + non_rwa)))
```

- [ ] **Step 4: Run the focused tests again**

Run: `uv run pytest tests/test_drivenmodal_coupled_postprocess.py -q --tb=short`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add squadds/simulations/drivenmodal/coupled_postprocess.py tests/test_drivenmodal_coupled_postprocess.py tests/fixtures/drivenmodal/README.md
git commit -m "feat: add drivenmodal coupled-system postprocessing"
```

---

### Task 8: Wire the new public API through `AnsysSimulator`

**Files:**
- Modify: `squadds/simulations/ansys_simulator.py`
- Modify: `tests/test_ansys_simulator.py`
- Create: `squadds/simulations/drivenmodal/hfss_runner.py`

- [ ] **Step 1: Write the failing facade tests**

```python
from squadds.simulations.ansys_simulator import AnsysSimulator
from squadds.simulations.drivenmodal.models import (
    CapacitanceExtractionRequest,
    DrivenModalLayerStackSpec,
    DrivenModalSweepSpec,
)


def test_ansys_simulator_exposes_run_drivenmodal(analyzer_stub, design_options_stub):
    simulator = AnsysSimulator(analyzer_stub, design_options_stub)
    request = CapacitanceExtractionRequest(
        system_kind="qubit_claw",
        design_payload={"design_options": {}},
        layer_stack=DrivenModalLayerStackSpec(),
        sweep=DrivenModalSweepSpec(start_ghz=1.0, stop_ghz=10.0, count=101),
    )
    assert hasattr(simulator, "run_drivenmodal")
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `uv run pytest tests/test_ansys_simulator.py -q --tb=short`
Expected: FAIL because `run_drivenmodal` does not exist

- [ ] **Step 3: Implement the minimal facade and runner seam**

```python
# squadds/simulations/drivenmodal/hfss_runner.py
def run_drivenmodal_request(request, *, checkpoint_dir=None, export_artifacts=True):
    return {
        "request": request,
        "checkpoint_dir": checkpoint_dir,
        "export_artifacts": export_artifacts,
    }
```

```python
# squadds/simulations/ansys_simulator.py
from squadds.simulations.drivenmodal.hfss_runner import run_drivenmodal_request


class AnsysSimulator:
    # ...
    def run_drivenmodal(self, request, *, checkpoint_dir=None, export_artifacts=True):
        return run_drivenmodal_request(
            request,
            checkpoint_dir=checkpoint_dir,
            export_artifacts=export_artifacts,
        )
```

- [ ] **Step 4: Run the focused tests again**

Run: `uv run pytest tests/test_ansys_simulator.py -q --tb=short`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add squadds/simulations/ansys_simulator.py squadds/simulations/drivenmodal/hfss_runner.py tests/test_ansys_simulator.py
git commit -m "feat: expose drivenmodal facade on ansys simulator"
```

---

### Task 9: Add dataset serialization helpers for summary rows and artifact URIs

**Files:**
- Create: `squadds/simulations/drivenmodal/dataset_records.py`
- Create: `tests/test_drivenmodal_dataset_records.py`

- [ ] **Step 1: Write the failing dataset-record tests**

```python
from squadds.simulations.drivenmodal.dataset_records import build_capacitance_summary_record


def test_build_capacitance_summary_record_includes_artifact_uris():
    record = build_capacitance_summary_record(
        run_id="demo-run",
        design={"design_options": {}},
        layer_stack=[{"material": "pec"}],
        sim_results={"capacitance_matrix_fF": [[1.0, -0.2], [-0.2, 0.9]]},
        artifacts={"touchstone_uri": "hf://datasets/SQuADDS/SQuADDS_DB/artifacts/drivenmodal/demo.s3p"},
    )
    assert record["artifacts"]["touchstone_uri"].startswith("hf://")
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `uv run pytest tests/test_drivenmodal_dataset_records.py -q --tb=short`
Expected: FAIL because the module does not exist

- [ ] **Step 3: Implement the minimal record builder**

```python
def build_capacitance_summary_record(*, run_id, design, layer_stack, sim_results, artifacts):
    return {
        "run_id": run_id,
        "design": design,
        "layer_stack": layer_stack,
        "sim_results": sim_results,
        "artifacts": artifacts,
        "provenance": {"simulation_family": "drivenmodal"},
    }
```

- [ ] **Step 4: Run the focused tests again**

Run: `uv run pytest tests/test_drivenmodal_dataset_records.py -q --tb=short`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add squadds/simulations/drivenmodal/dataset_records.py tests/test_drivenmodal_dataset_records.py
git commit -m "feat: add drivenmodal dataset record builders"
```

---

### Task 10: Add the tutorials and documentation updates

**Files:**
- Create: `docs/source/tutorials/Tutorial-10_DrivenModal_Capacitance_Extraction.ipynb`
- Create: `docs/source/tutorials/Tutorial-11_DrivenModal_Coupled_System_Postprocessing.ipynb`
- Modify: `docs/source/tutorials/index.rst`
- Modify: `README.md`
- Modify: `docs/source/release_notes.rst`

- [ ] **Step 1: Write the docs-first failing check**

```python
def test_tutorial_filenames_are_registered():
    from pathlib import Path

    index_text = Path("docs/source/tutorials/index.rst").read_text()
    assert "Tutorial-10_DrivenModal_Capacitance_Extraction.ipynb" in index_text
    assert "Tutorial-11_DrivenModal_Coupled_System_Postprocessing.ipynb" in index_text
```

- [ ] **Step 2: Run the focused docs test to verify it fails**

Run: `uv run pytest tests/test_drivenmodal_docs.py -q --tb=short`
Expected: FAIL because the notebook files and registration do not exist

- [ ] **Step 3: Add the tutorial shells and index links**

```rst
.. toctree::
   :maxdepth: 1

   Tutorial-10_DrivenModal_Capacitance_Extraction.ipynb
   Tutorial-11_DrivenModal_Coupled_System_Postprocessing.ipynb
```

```markdown
# Tutorial requirements

- Tutorial 10 compares driven-modal qubit-claw and NCap capacitance extraction against existing Q3D-backed values in SQuADDS.
- Tutorial 11 compares driven-modal coupled-system post-processing against the pre-simulated SQuADDS record.
- Both tutorials end with a section explaining how the resulting datasets will be structured and what the public API exposes.
```

- [ ] **Step 4: Run the focused docs test again**

Run: `uv run pytest tests/test_drivenmodal_docs.py -q --tb=short`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add docs/source/tutorials/index.rst docs/source/tutorials/Tutorial-10_DrivenModal_Capacitance_Extraction.ipynb docs/source/tutorials/Tutorial-11_DrivenModal_Coupled_System_Postprocessing.ipynb README.md docs/source/release_notes.rst tests/test_drivenmodal_docs.py
git commit -m "docs: add drivenmodal tutorials and release notes"
```

---

### Task 11: Run the complete verification suite for the implementation wave

**Files:**
- Modify: any files required to fix issues discovered during verification

- [ ] **Step 1: Run focused driven-modal tests**

Run: `uv run pytest tests/test_drivenmodal_models.py tests/test_drivenmodal_layer_stack.py tests/test_drivenmodal_ports.py tests/test_drivenmodal_artifacts.py tests/test_drivenmodal_capacitance.py tests/test_drivenmodal_coupled_postprocess.py tests/test_drivenmodal_dataset_records.py tests/test_ansys_simulator.py -q --tb=short`
Expected: PASS

- [ ] **Step 2: Run the broader unit suite**

Run: `uv run pytest -q --tb=short`
Expected: PASS, with only pre-existing skips allowed

- [ ] **Step 3: Run formatting and lint checks**

Run: `uv run --extra dev ruff check squadds/simulations/drivenmodal squadds/simulations/ansys_simulator.py tests`
Expected: PASS

Run: `uv run --extra dev ruff format --check squadds/simulations/drivenmodal squadds/simulations/ansys_simulator.py tests`
Expected: PASS

- [ ] **Step 4: Build docs**

Run: `cd docs && uv run make html`
Expected: PASS with the two new tutorials rendered and linked

- [ ] **Step 5: Commit the verification fixes**

```bash
git add squadds/simulations/drivenmodal squadds/simulations/ansys_simulator.py tests docs/source/tutorials README.md docs/source/release_notes.rst pyproject.toml
git commit -m "test: verify drivenmodal implementation and docs"
```

---

## Self-Review

### Spec coverage

- Public API: covered in Tasks 1, 2, and 8
- Layer stack: covered in Task 3
- Port handling: covered in Task 4
- Checkpointing and artifact manifests: covered in Task 5
- Capacitance extraction: covered in Task 6
- Coupled-system post-processing: covered in Task 7
- Dataset contract: covered in Task 9
- Tutorials and docs: covered in Task 10
- Verification: covered in Task 11

### Placeholder scan

- No placeholder markers remain.
- All code-modifying tasks include example code or exact commands.

### Type consistency

- Request model names are consistent with the PRD:
  - `DrivenModalLayerStackSpec`
  - `CapacitanceExtractionRequest`
  - `CoupledSystemDrivenModalRequest`
- The public facade method is consistently named `run_drivenmodal`.
