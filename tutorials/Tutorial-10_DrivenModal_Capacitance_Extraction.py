# %% [markdown]
# # Tutorial 10: Driven-Modal Capacitance Extraction
#
# In this tutorial we will use SQuADDS to set up a driven-modal HFSS
# capacitance extraction for two familiar objects:
#
# 1. a `TransmonCross` qubit-claw capacitor, and
# 2. an `NCap` interdigital coupler.
#
# The goal is not to teach every Ansys scripting detail. SQuADDS should carry
# that weight. Instead, this notebook teaches the SQuADDS contract for this
# workflow: a validated database row becomes a portable driven-modal request,
# the request renders the same physical network into HFSS, and the resulting
# Y-parameters are converted back into the capacitance quantities users already
# know from the Q3D datasets.
#
# By the end, you should know how to:
#
# - pull a Q3D reference row from the SQuADDS database,
# - build the equivalent driven-modal request,
# - compare driven-modal capacitances against the Q3D reference, and
# - read the Maxwell capacitance matrix without double-counting terms.

# %% [markdown]
# ## Imports
#
# We will use the regular SQuADDS database API to choose designs, then the
# driven-modal workflow helpers to build the HFSS requests.

# %%
from pathlib import Path

import pandas as pd

from squadds import SQuADDS_DB
from squadds.simulations.drivenmodal import (
    build_capacitance_request,
    capacitance_comparison_table,
    capacitance_reference_summary,
    default_capacitance_setup,
    default_capacitance_sweep,
    maxwell_matrix_interpretation,
)

try:
    from IPython.display import display
except ImportError:  # pragma: no cover

    def display(obj):
        print(obj)


# %% [markdown]
# ## Choose Reference Designs
#
# We start from already-validated SQuADDS rows. This is useful because the Q3D
# capacitance matrix in each row gives us a trusted comparison target.
#
# `see_dataset(...)` returns the raw table behind one SQuADDS dataset. The
# important inputs are:
#
# - `component`: the high-level object family, for example `"qubit"` or
#   `"coupler"`.
# - `component_name`: the concrete Qiskit Metal component stored in the dataset.
# - `data_type`: the simulated observable we want from the database. Here it is
#   `"cap_matrix"` because we want Q3D capacitances for comparison.
#
# In a real study you would usually pick the row returned by an inverse-design
# query. For a tutorial, choosing fixed rows keeps the notebook deterministic.

# %%
db = SQuADDS_DB()

qubit_df = db.see_dataset(
    component="qubit",
    component_name="TransmonCross",
    data_type="cap_matrix",
)
ncap_df = db.see_dataset(
    component="coupler",
    component_name="NCap",
    data_type="cap_matrix",
)

qubit_row = qubit_df.iloc[3]
ncap_row = ncap_df.iloc[1]

display(qubit_row[["design_options", "cross_to_ground", "cross_to_claw"]])
display(ncap_row[["design_options", "top_to_bottom", "top_to_ground"]])


# %% [markdown]
# ## Declare the Driven-Modal Solve
#
# A driven-modal capacitance extraction is a frequency-domain multiport solve.
# SQuADDS wraps the bookkeeping in a `CapacitanceExtractionRequest`.
#
# Behind the scenes, `build_capacitance_request(...)` does four things:
#
# - It reads the selected row's `design_options` and keeps the geometry tied to
#   the database reference.
# - It declares which Qiskit Metal pins become HFSS lumped ports. For the
#   qubit-claw case these are the transmon cross/JJ node and the readout claw;
#   for the NCap case these are the top and bottom capacitor terminals.
# - It attaches a SQuADDS layer-stack preset where metal is rendered as PEC and
#   the substrate metadata matches the Ansys renderer expectations.
# - It stores the HFSS adaptive setup, frequency sweep, and artifact policy in a
#   serializable object that can be sent to the Windows Ansys machine.
#
# The only choices we make here are the ones a user should care about:
#
# - which geometry row to render,
# - which physical ports define the capacitance network,
# - which setup/sweep settings to use, and
# - where checkpointed artifacts should be stored.
#
# `default_capacitance_setup(freq_ghz=5.0)` is the adaptive HFSS setup. The
# frequency is not "the extracted capacitance frequency"; it is the frequency at
# which HFSS builds the adaptive mesh. The default uses `max_delta_s=0.005`,
# `min_converged=5`, and mixed basis order because those settings gave stable
# agreement with the corresponding Q3D workflow.
#
# `default_capacitance_sweep(...)` is the frequency grid where HFSS exports the
# network data. The 1-10 GHz interpolating sweep is broad enough to catch
# frequency dependence while still being cheap enough for a validation pass.

# %%
setup = default_capacitance_setup(freq_ghz=5.0)
sweep = default_capacitance_sweep(start_ghz=1.0, stop_ghz=10.0, count=400)

qubit_request = build_capacitance_request(
    qubit_row,
    system_kind="qubit_claw",
    run_id="tutorial10-qubit-claw",
    setup=setup,
    sweep=sweep,
)
ncap_request = build_capacitance_request(
    ncap_row,
    system_kind="ncap",
    run_id="tutorial10-ncap",
    setup=setup,
    sweep=sweep,
)

display(pd.DataFrame([qubit_request.to_dict(), ncap_request.to_dict()]))


# %% [markdown]
# ## Run on the Ansys Machine
#
# The request object is the portable contract. On the Windows machine with HFSS
# installed, pass it to the driven-modal runner and keep the generated runtime
# folder. The notebook can then be re-opened later and the post-processing can
# be repeated without re-solving.
#
# `run_drivenmodal_request(...)` is intentionally the only Ansys-facing API used
# here. It renders the Qiskit Metal geometry, creates the HFSS driven-modal
# setup, assigns the lumped ports from the request, exports Touchstone/Y data,
# and writes checkpoint files. If a run is interrupted, the artifact policy can
# resume from completed stages instead of starting from zero.
#
# The cell below is intentionally guarded. Leave `RUN_ANSYS = False` when you
# are reading the tutorial on a laptop or in CI.

# %%
RUN_ANSYS = False
CHECKPOINT_ROOT = Path("tutorials/runtime/drivenmodal_capacitance/checkpoints")

if RUN_ANSYS:
    from squadds.simulations.drivenmodal.hfss_runner import run_drivenmodal_request

    qubit_prepared = run_drivenmodal_request(qubit_request, checkpoint_dir=CHECKPOINT_ROOT)
    ncap_prepared = run_drivenmodal_request(ncap_request, checkpoint_dir=CHECKPOINT_ROOT)

    print("Qubit-claw run directory:", qubit_prepared["manifest"]["run_dir"])
    print("NCap run directory:", ncap_prepared["manifest"]["run_dir"])


# %% [markdown]
# ## Compare Against Q3D
#
# After the solve, the driven-modal post-processing writes a compact summary
# dictionary. The comparison table below is the only table a typical user needs
# to inspect.
#
# The conversion used by the helper is the small-signal capacitor relation:
#
# $$Y(\omega) = j \omega C.$$
#
# Therefore SQuADDS computes the active-node capacitance matrix from the
# imaginary part of the exported admittance matrix:
#
# $$C(\omega) = \frac{\operatorname{Im}(Y(\omega))}{\omega}.$$
#
# The example below uses the Q3D row as a stand-in so the docsite can show the
# table without launching HFSS. In an actual run, replace
# `example_qubit_drivenmodal` and `example_ncap_drivenmodal` with
# `summary["drivenmodal_summary_fF"]` from the completed checkpoint.

# %%
qubit_q3d = capacitance_reference_summary(qubit_row, system_kind="qubit_claw")
ncap_q3d = capacitance_reference_summary(ncap_row, system_kind="ncap")

# Replace these with `summary["drivenmodal_summary_fF"]` from a completed run.
example_qubit_drivenmodal = qubit_q3d
example_ncap_drivenmodal = ncap_q3d

display(capacitance_comparison_table(drivenmodal_fF=example_qubit_drivenmodal, q3d_fF=qubit_q3d))
display(capacitance_comparison_table(drivenmodal_fF=example_ncap_drivenmodal, q3d_fF=ncap_q3d))


# %% [markdown]
# ## Reading the Matrix Correctly
#
# A common source of confusion is the Maxwell capacitance matrix convention.
# The diagonal entries are not extra capacitances to add on top of all pair
# terms. For a two-node qubit-claw model, the transmon shunt capacitance is:
#
# $$C_\Sigma = C_{\text{cross-ground}} + C_{\text{cross-claw}}.$$
#
# not:
#
# $$C_{\text{cross-cross}} + C_{\text{cross-ground}} + C_{\text{cross-claw}}.$$
#
# SQuADDS' Hamiltonian extraction uses the first expression, so the mutual
# capacitance is not double-counted.

# %%
display(maxwell_matrix_interpretation())


# %% [markdown]
# ## What to Check Before Trusting a Run
#
# Before comparing numbers, inspect these three artifacts from the runtime
# folder:
#
# 1. `layer_stack.csv`: confirms cryogenic silicon and metal-layer choices.
# 2. `qiskit_layout.png` or the saved HFSS screenshot: confirms the rendered
#    geometry and ports match the intended device.
# 3. `comparison.csv`: confirms the extracted capacitance matrix is being
#    compared to the corresponding Q3D row.
#
# If those three checks are clean, the capacitance comparison table is the
# right high-level summary to report.

# %% [markdown]
# ## License
# <div style='width: 100%; background-color:#3cb1c2;color:#324344;padding-left: 10px; padding-bottom: 10px; padding-right: 10px; padding-top: 5px'>
#     <h3>This code is a part of SQuADDS</h3>
#     <p>Developed by Sadman Ahmed Shanto</p>
#     <p>&copy; Copyright 2023.</p>
# </div>
