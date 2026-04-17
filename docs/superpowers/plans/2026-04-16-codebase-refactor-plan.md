# SQuADDS Codebase Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor SQuADDS into smaller, better-tested internal modules while preserving current public behavior and compatibility.

**Architecture:** Keep `SQuADDS_DB`, `Analyzer`, simulator entrypoints, and existing dataset JSON contracts as stable compatibility facades. Move behavior-preserving internals into smaller focused modules with characterization tests wrapped around the current behavior before each extraction.

**Tech Stack:** Python 3.10+, `uv`, `pytest`, `datasets`, `huggingface_hub`, `scipy`, `qiskit-metal`, Streamlit

---

## File Structure

- Modify: `/.gitignore`
- Modify: `/pyproject.toml`
- Modify: `/tests/imports_test.py`
- Modify: `/tests/mvp_test.py`
- Create: `/tests/conftest.py`
- Create: `/tests/test_sweeper_helperfunctions.py`
- Modify: `/squadds/simulations/sweeper_helperfunctions.py`
- Create: `/squadds/core/db_catalog.py`
- Create: `/squadds/core/db_loader.py`
- Create: `/squadds/core/db_selection.py`
- Create: `/squadds/core/db_merge.py`
- Modify: `/squadds/core/db.py`
- Create: `/squadds/core/analysis_enrichment.py`
- Create: `/squadds/core/analysis_search.py`
- Create: `/squadds/core/analysis_plotting.py`
- Modify: `/squadds/core/analysis.py`
- Create: `/squadds/simulations/sweep_plan.py`
- Create: `/squadds/simulations/result_normalization.py`
- Create: `/squadds/database/contributor_env.py`
- Create: `/squadds/database/hf_dataset_ops.py`
- Create: `/squadds/database/github_ops.py`
- Create: `/docs/superpowers/specs/2026-04-16-codebase-refactor-design.md`
- Create: `/docs/superpowers/plans/2026-04-16-codebase-refactor-plan.md`

### Task 1: Establish Repo Hygiene and Characterization Tests

**Files:**
- Modify: `/.gitignore`
- Modify: `/pyproject.toml`
- Modify: `/tests/imports_test.py`
- Modify: `/tests/mvp_test.py`
- Create: `/tests/conftest.py`
- Create: `/tests/test_sweeper_helperfunctions.py`
- Test: `/tests/imports_test.py`, `/tests/test_sweeper_helperfunctions.py`

- [ ] **Step 1: Turn the import smoke script into real pytest tests**

```python
import importlib
import pytest

MODULES = [
    "squadds",
    "squadds.calcs",
    "squadds.core",
    "squadds.database",
    "squadds.interpolations",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name):
    importlib.import_module(module_name)
```

- [ ] **Step 2: Add characterization tests for sweep helper behavior**

```python
def test_extract_qsweep_parameters_builds_every_nested_combination():
    sweep = {
        "cplr_opts": {"finger_count": [1, 2]},
        "claw_opts": {"connection_pads": {"readout": {"claw_length": ["10um", "20um"]}}},
    }

    combos = extract_QSweep_parameters(sweep)

    assert combos == [
        {"cplr_opts": {"finger_count": 1}, "claw_opts": {"connection_pads": {"readout": {"claw_length": "10um"}}}},
        {"cplr_opts": {"finger_count": 1}, "claw_opts": {"connection_pads": {"readout": {"claw_length": "20um"}}}},
        {"cplr_opts": {"finger_count": 2}, "claw_opts": {"connection_pads": {"readout": {"claw_length": "10um"}}}},
        {"cplr_opts": {"finger_count": 2}, "claw_opts": {"connection_pads": {"readout": {"claw_length": "20um"}}}},
    ]
```

- [ ] **Step 3: Gate live/network/simulator tests behind an explicit env flag**

```python
RUN_LIVE = os.getenv("SQUADDS_RUN_LIVE_TESTS") == "1"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not RUN_LIVE, reason="Set SQUADDS_RUN_LIVE_TESTS=1 to run live integration tests."),
]
```

- [ ] **Step 4: Run the fast default test suite**

Run: `uv run --extra dev pytest -q tests/imports_test.py tests/test_sweeper_helperfunctions.py`
Expected: PASS with all fast tests green

- [ ] **Step 5: Commit the foundation pass**

```bash
git add .gitignore pyproject.toml tests/imports_test.py tests/mvp_test.py tests/conftest.py tests/test_sweeper_helperfunctions.py
git commit -m "test: establish characterization baseline for refactor work"
```

### Task 2: Refactor Sweep Helper Internals Without Changing Behavior

**Files:**
- Modify: `/squadds/simulations/sweeper_helperfunctions.py`
- Test: `/tests/test_sweeper_helperfunctions.py`

- [ ] **Step 1: Refactor the module behind the new characterization tests**

```python
def extract_QSweep_parameters(parameters: dict) -> list[dict]:
    flattened_keys = extract_parameters(parameters)
    flattened_values = extract_values(parameters)
    combinations = generate_combinations(flattened_values)
    return create_dict_list(flattened_keys, combinations)
```

```python
def _assign_nested_value(target: dict, dotted_key: str, value):
    current = target
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value
```

- [ ] **Step 2: Re-run the focused test suite**

Run: `uv run --extra dev pytest -q tests/test_sweeper_helperfunctions.py tests/imports_test.py`
Expected: PASS with no behavior regressions

- [ ] **Step 3: Commit the sweep-helper refactor**

```bash
git add squadds/simulations/sweeper_helperfunctions.py tests/test_sweeper_helperfunctions.py tests/imports_test.py
git commit -m "refactor: simplify sweep helper internals under characterization tests"
```

### Task 3: Decompose `SQuADDS_DB` into Focused Internal Modules

**Files:**
- Create: `/squadds/core/db_catalog.py`
- Create: `/squadds/core/db_loader.py`
- Create: `/squadds/core/db_selection.py`
- Create: `/squadds/core/db_merge.py`
- Modify: `/squadds/core/db.py`
- Test: `/tests/test_db_catalog.py`, `/tests/test_db_selection.py`, `/tests/test_db_merge.py`

- [ ] **Step 1: Characterize current selection and merge behavior before extraction**

```python
def test_select_resonator_type_maps_to_expected_coupler():
    db = SQuADDS_DB()
    db.select_system(["qubit", "cavity_claw"])
    db.select_resonator_type("half")
    assert db.selected_coupler == "NCap"
```

- [ ] **Step 2: Extract config discovery and dataset-loading helpers**

```python
def load_supported_config_names(repo_name: str) -> list[str]:
    configs = get_dataset_config_names(repo_name, download_mode="force_redownload")
    return [config for config in configs if config.count("-") == 2]
```

- [ ] **Step 3: Extract dataframe merge helpers and keep `SQuADDS_DB` as the facade**

```python
class SQuADDS_DB(...):
    def create_system_df(self, parallelize=False, num_cpu=None):
        if isinstance(self.selected_system, str):
            return create_single_component_df(...)
        return create_multi_component_df(...)
```

- [ ] **Step 4: Run targeted DB tests**

Run: `uv run --extra dev pytest -q tests/test_db_catalog.py tests/test_db_selection.py tests/test_db_merge.py`
Expected: PASS with current DB behavior preserved

- [ ] **Step 5: Commit the DB decomposition**

```bash
git add squadds/core/db.py squadds/core/db_catalog.py squadds/core/db_loader.py squadds/core/db_selection.py squadds/core/db_merge.py tests/test_db_catalog.py tests/test_db_selection.py tests/test_db_merge.py
git commit -m "refactor: split database catalog, selection, and merge logic"
```

### Task 4: Decompose `Analyzer` into Search, Enrichment, and Plotting Helpers

**Files:**
- Create: `/squadds/core/analysis_enrichment.py`
- Create: `/squadds/core/analysis_search.py`
- Create: `/squadds/core/analysis_plotting.py`
- Modify: `/squadds/core/analysis.py`
- Test: `/tests/test_analysis_search.py`, `/tests/test_analysis_plotting.py`

- [ ] **Step 1: Characterize `find_closest()` and H-space plotting behavior**

```python
def test_find_closest_preserves_lowest_distance_first():
    analyzer = Analyzer(db)
    result = analyzer.find_closest(target_params=target_params, num_top=1)
    assert len(result) == 1
```

- [ ] **Step 2: Extract pure search code from the stateful facade**

```python
def find_closest_rows(df, target_params, metric_strategy, num_top):
    distances = metric_strategy.calculate_vectorized(target_params, df)
    return df.loc[distances.nsmallest(num_top).index]
```

- [ ] **Step 3: Extract plotting and result-shaping helpers**

```python
def build_quarter_wave_hspace_plot(df, target_params, closest_row):
    ...
```

- [ ] **Step 4: Run targeted analysis tests**

Run: `uv run --extra dev pytest -q tests/test_analysis_search.py tests/test_analysis_plotting.py`
Expected: PASS with stable analyzer behavior

- [ ] **Step 5: Commit the analyzer decomposition**

```bash
git add squadds/core/analysis.py squadds/core/analysis_enrichment.py squadds/core/analysis_search.py squadds/core/analysis_plotting.py tests/test_analysis_search.py tests/test_analysis_plotting.py
git commit -m "refactor: split analyzer enrichment, search, and plotting concerns"
```

### Task 5: Normalize Simulation and Contribution Workflow Boundaries

**Files:**
- Create: `/squadds/simulations/sweep_plan.py`
- Create: `/squadds/simulations/result_normalization.py`
- Create: `/squadds/database/contributor_env.py`
- Create: `/squadds/database/hf_dataset_ops.py`
- Create: `/squadds/database/github_ops.py`
- Modify: `/squadds/simulations/objects.py`
- Modify: `/squadds/database/contributor.py`
- Modify: `/squadds/database/contributor_HF.py`
- Modify: `/squadds/database/github.py`
- Test: `/tests/test_result_normalization.py`, `/tests/test_contributor_env.py`

- [ ] **Step 1: Characterize current payload normalization and env lookup behavior**

```python
def test_cap_matrix_normalization_preserves_existing_key_names():
    normalized = normalize_cap_matrix_payload(raw_payload)
    assert normalized["sim_results"]["top_to_top"] == raw_payload["sim_results"]["C_top2top"]
```

- [ ] **Step 2: Extract pure normalization helpers from simulator code**

```python
def normalize_cap_matrix_result(raw_result: dict) -> dict:
    ...
```

- [ ] **Step 3: Extract env/config access from contributor classes**

```python
def load_contributor_env(env_path=ENV_FILE_PATH) -> dict[str, str | None]:
    ...
```

- [ ] **Step 4: Run the targeted workflow tests**

Run: `uv run --extra dev pytest -q tests/test_result_normalization.py tests/test_contributor_env.py`
Expected: PASS with behavior preserved

- [ ] **Step 5: Commit the workflow cleanup**

```bash
git add squadds/simulations/objects.py squadds/simulations/sweep_plan.py squadds/simulations/result_normalization.py squadds/database/contributor.py squadds/database/contributor_HF.py squadds/database/github.py squadds/database/contributor_env.py squadds/database/hf_dataset_ops.py squadds/database/github_ops.py tests/test_result_normalization.py tests/test_contributor_env.py
git commit -m "refactor: isolate simulation normalization and contribution services"
```

### Task 6: Thin the UI Layer and Align Docs with the Refactor

**Files:**
- Modify: `/squadds/ui/app.py`
- Modify: `/squadds/ui/utils_query.py`
- Modify: `/README.md`
- Modify: `/docs/source/getting_started.rst`

- [ ] **Step 1: Extract query-orchestration helpers that the UI currently owns inline**

```python
def run_design_search(system_type, qubit_type, cavity_type, resonator_type, params, num_results, num_cpu, skip_df_gen):
    ...
```

- [ ] **Step 2: Keep the Streamlit file focused on session state and rendering**

```python
def main():
    initialize_session_state()
    render_sidebar()
    render_results()
```

- [ ] **Step 3: Update docs to reflect the stable public entrypoints after the internal refactor**

Run: `uv run --extra dev pytest -q tests/imports_test.py`
Expected: PASS with public API still importable

- [ ] **Step 4: Commit the UI/doc alignment**

```bash
git add squadds/ui/app.py squadds/ui/utils_query.py README.md docs/source/getting_started.rst
git commit -m "refactor: thin UI orchestration and align docs with stable facades"
```
