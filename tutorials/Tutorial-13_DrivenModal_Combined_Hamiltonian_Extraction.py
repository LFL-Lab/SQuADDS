# %% [markdown]
# # Tutorial 13: Unified Driven-Modal Hamiltonian Extraction
#
# In Tutorial 10 we used driven-modal HFSS to reproduce capacitance-style
# quantities. Here we use the same idea for a complete qubit-cavity system.
#
# The workflow is:
#
# 1. ask SQuADDS for a qubit-cavity design near a target Hamiltonian,
# 2. render that one coupled geometry into HFSS,
# 3. sweep finely around the qubit band and the resonator band,
# 4. extract \(f_q\), \(\alpha\), \(f_r\), \(\kappa\), \(\chi\), and \(g\), and
# 5. compare the driven-modal result against the SQuADDS reference row.
#
# The docsite labels this as Tutorial 11, because the intermediate development
# notebooks are intentionally not part of the public tutorial sequence.

# %% [markdown]
# ## Imports
#
# The important point is that the notebook uses SQuADDS-level objects. We do
# not build Qiskit Metal components or Ansys COM calls by hand here.

# %%
from pathlib import Path

import pandas as pd

from squadds import Analyzer, SQuADDS_DB
from squadds.simulations.drivenmodal import (
    build_segmented_coupled_system_requests,
    coupled_reference_summary,
    default_hamiltonian_setup,
    hamiltonian_comparison_table,
    segmented_hamiltonian_sweeps,
)

try:
    from IPython.display import display
except ImportError:  # pragma: no cover

    def display(obj):
        print(obj)


# %% [markdown]
# ## Query a Target Design
#
# This follows the same pattern as Tutorial 1 and Tutorial 5: select a system,
# choose target Hamiltonian parameters, and let the `Analyzer` return the best
# matching SQuADDS design.

# %%
db = SQuADDS_DB()
db.select_system(["qubit", "cavity_claw"])
db.select_qubit("TransmonCross")
db.select_cavity_claw("RouteMeander")
db.select_resonator_type("quarter")

system_df = db.create_system_df()
analyzer = Analyzer(db)

target_params = {
    "qubit_frequency_GHz": 4.0,
    "anharmonicity_MHz": -130,
    "cavity_frequency_GHz": 8.9,
    "kappa_kHz": 300,
    "g_MHz": 50,
    "resonator_type": "quarter",
}

results = analyzer.find_closest(target_params=target_params, num_top=3, metric="Euclidean")
reference_row = results.iloc[0]

display(system_df.head())
display(results.head(3))


# %% [markdown]
# ## Build the Driven-Modal Requests
#
# A coupled-system driven-modal solve is a 3-port network:
#
# 1. feedline input,
# 2. feedline output, and
# 3. the Josephson junction port.
#
# We use the same geometry for all sweeps. The qubit and resonator bands are
# dense because that is where the physics lives. The bridge sweep is coarse
# because it only connects the two frequency windows.

# %%
reference = coupled_reference_summary(reference_row)

setup = default_hamiltonian_setup(freq_ghz=reference["cavity_frequency_ghz"])
sweeps = segmented_hamiltonian_sweeps(
    reference,
    qubit_count=22_000,
    bridge_count=4_001,
    resonator_count=22_000,
)
requests = build_segmented_coupled_system_requests(
    reference_row,
    resonator_type="quarter",
    run_id="tutorial11-quarter-wave",
    reference=reference,
    setup=setup,
    sweeps=sweeps,
)

sweep_table = pd.DataFrame(
    [
        {
            "band": name,
            "start_GHz": request.sweep.start_ghz,
            "stop_GHz": request.sweep.stop_ghz,
            "points": request.sweep.count,
            "type": request.sweep.sweep_type,
        }
        for name, request in requests.items()
    ]
)
display(sweep_table)


# %% [markdown]
# ## Run on the Ansys Machine
#
# The solve cell is guarded because a production sweep can take hours. On the
# lab machine, set `RUN_ANSYS = True`. SQuADDS will write checkpointed artifacts
# so post-processing can be repeated locally.

# %%
RUN_ANSYS = False
CHECKPOINT_ROOT = Path("tutorials/runtime/drivenmodal_combined_hamiltonian/checkpoints")

if RUN_ANSYS:
    from squadds.simulations.drivenmodal.hfss_runner import run_drivenmodal_request

    prepared_runs = {
        band: run_drivenmodal_request(request, checkpoint_dir=CHECKPOINT_ROOT) for band, request in requests.items()
    }
    for band, prepared in prepared_runs.items():
        print(f"{band}: {prepared['manifest']['run_dir']}")


# %% [markdown]
# ## Post-Process the Same Network in Two Ways
#
# The HFSS output is a frequency-dependent 3-port admittance matrix.
#
# Around the qubit band, SQuADDS reduces the network to the JJ port and uses
# `scqubits` to convert the linearized mode into \(f_q\) and \(\alpha\).
#
# Around the resonator band, SQuADDS terminates the JJ port with the
# state-dependent junction inductances and reads the loaded feedline response.
# The two loaded resonances give \(\chi\). Then \(g\) is recovered from the
# dispersive transmon relation.

# %%
# Replace this dictionary with the summary produced by a completed run.
example_drivenmodal = {
    "qubit_frequency_ghz": reference["qubit_frequency_ghz"],
    "anharmonicity_mhz": reference["anharmonicity_mhz"],
    "cavity_frequency_ghz": reference["cavity_frequency_ghz"],
    "kappa_mhz": reference["kappa_mhz"],
    "g_mhz": reference["g_mhz"],
    "chi_mhz": float("nan"),
}

display(hamiltonian_comparison_table(drivenmodal=example_drivenmodal, squadds=reference))


# %% [markdown]
# ## What the Comparison Means
#
# The SQuADDS row is the pre-simulated reference. The driven-modal row is a
# full-wave re-simulation of the same geometry with a different electromagnetic
# observable.
#
# A good result should satisfy three sanity checks:
#
# 1. the rendered HFSS geometry includes the qubit, claw, resonator, and
#    feedline ports,
# 2. silicon is set to the cryogenic value used by the Q3D/eigenmode flows, and
# 3. the Hamiltonian table agrees within the accuracy expected from the sweep
#    resolution and the chosen JJ termination model.
#
# If the resonator frequency shifts below the bare SQuADDS cavity frequency,
# that is not automatically a bug: the driven-modal model includes capacitive
# loading from the qubit.

# %% [markdown]
# ## License
# <div style='width: 100%; background-color:#3cb1c2;color:#324344;padding-left: 10px; padding-bottom: 10px; padding-right: 10px; padding-top: 5px'>
#     <h3>This code is a part of SQuADDS</h3>
#     <p>Developed by Sadman Ahmed Shanto</p>
#     <p>&copy; Copyright 2023.</p>
# </div>
