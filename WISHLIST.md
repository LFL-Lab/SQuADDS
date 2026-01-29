# SQuADDS Feature Wishlist

## High Priority

### 1. Ansys-Native Parallel Simulation
**Status:** Not Implemented
**Priority:** High
**Description:** Implement true Ansys-native parallel simulation where multiple simulations (Eigenmode, Qubit LOM, Coupler LOM) run concurrently within the same Ansys project.

**Implementation Plan:**
- Render all designs to the same Ansys project
- Start all analyses using Ansys's built-in scheduling
- Poll for completion with progress updates
- Post-process results when all simulations complete

**Benefits:**
- Leverage Ansys license-based parallelization
- Reduce total simulation time for coupled systems
- Better resource utilization

**Technical Challenges:**
- Need to separate rendering from analysis execution
- Implement polling mechanism for completion status
- Handle different simulation completion times
- Ensure proper error handling for partial failures

**Related Files:**
- `squadds/simulations/objects.py` - `simulate_whole_device()`, `run_eigenmode()`, `run_xmon_LOM()`, `run_capn_LOM()`
- `squadds/simulations/ansys_simulator.py` - `simulate()` method
- `squadds/simulations/utils.py` - Result extraction functions

**GitHub Issue:** TBD

---

## Medium Priority

### 2. PyAEDT Version Compatibility
**Status:** Under Investigation
**Priority:** Medium
**Description:** Resolve compatibility issues between Qiskit Metal (using pyAEDT 0.23.0) and pyEPR workflows that previously worked with pyAEDT 0.6.x.

**Symptoms:**
- gRPC errors during simulation
- Breaking changes in pyAEDT API between versions

**Investigation Needed:**
- Identify specific API changes in pyAEDT 0.6.x → 0.23.0
- Test pyEPR compatibility with newer pyAEDT versions
- Determine if Qiskit Metal wrapper needs updates

---

## Low Priority

### 3. Enhanced Visualization Options
**Status:** Idea
**Priority:** Low
**Description:** Add more visualization options for simulation results and design parameters.

**Ideas:**
- Interactive 3D plots of designs
- Animated convergence visualization
- Comparison plots for multiple designs

---

## Completed Features

### ✅ Granular Simulation Setup Control
**Completed:** v0.4.3
**Description:** Added `update_simulation_setup(target, **kwargs)` for granular control over simulation parameters with intelligent target mapping.

### ✅ Setup Viewing API
**Completed:** v0.4.3
**Description:** Added `get_simulation_setup(target)` to view current simulation parameters in formatted tables.

### ✅ Half-Wave Cavity Support
**Completed:** v0.4.3
**Description:** All three setups (`setup_qubit`, `setup_cavity_claw`, `setup_coupler`) properly included in HWC dataframes.

### ✅ Plot Generation Control
**Completed:** v0.4.3
**Description:** Added `generate_plots` flag to disable plot generation by default, significantly speeding up simulations.
