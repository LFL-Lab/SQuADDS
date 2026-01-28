import os

import matplotlib as mpl

# Check if a display is available
if os.environ.get("DISPLAY", "") == "":
    print("No display found. Running in headless mode.")
    mpl.use("Agg")  # Set the matplotlib backend to Agg (headless)
    os.environ["QISKIT_METAL_HEADLESS"] = "1"  # Set Qiskit Metal to headless mode
else:
    print("Display found. Running with GUI support.")
    # You can let matplotlib choose the default backend (Qt5Agg, TkAgg, etc.)
    if "QISKIT_METAL_HEADLESS" in os.environ:
        del os.environ["QISKIT_METAL_HEADLESS"]  # Remove the headless flag if GUI is available


from squadds import Analyzer, SQuADDS_DB
from squadds.interpolations.physics import ScalingInterpolator

db = SQuADDS_DB()
db.view_datasets()
db.get_configs()
db.get_dataset_info(component="qubit", component_name="TransmonCross", data_type="cap_matrix")
db.get_dataset_info(component="cavity_claw", component_name="RouteMeander", data_type="eigenmode")
db.see_dataset(component="qubit", component_name="TransmonCross", data_type="cap_matrix")
db.view_all_contributors()

db.unselect_all()
db.select_system(["qubit", "cavity_claw"])
db.select_qubit("TransmonCross")
db.select_cavity_claw("RouteMeander")
db.select_resonator_type("quarter")
db.show_selections()
merged_df = db.create_system_df()
print(merged_df)

analyzer = Analyzer(db)

target_params = {
    "qubit_frequency_GHz": 4,
    "cavity_frequency_GHz": 6.2,
    "kappa_kHz": 120,
    "resonator_type": "quarter",
    "anharmonicity_MHz": -200,
    "g_MHz": 70,
}

results = analyzer.find_closest(target_params=target_params, num_top=1, metric="Euclidean", display=True)
print(results)

interpolator = ScalingInterpolator(analyzer, target_params)

design_df = interpolator.get_design()

print(design_df)

print("=" * 50)
print("Testing Half-Wave Cavity (HWC)...")
db.unselect_all()
db.select_system(["qubit", "cavity_claw"])
db.select_qubit("TransmonCross")
db.select_cavity_claw("RouteMeander")
db.select_resonator_type("half")
db.show_selections()
merged_df = db.create_system_df()

target_params_hwc = {
    "qubit_frequency_GHz": 4,
    "cavity_frequency_GHz": 6.2,
    "kappa_kHz": 120,
    "anharmonicity_MHz": -200,
    "g_MHz": 70,
}

results_hwc = analyzer.find_closest(target_params=target_params_hwc, num_top=1, metric="Euclidean", display=True)
print(results_hwc)

interpolator_hwc = ScalingInterpolator(analyzer, target_params_hwc)
design_df_hwc = interpolator_hwc.get_design()
print(design_df_hwc)
print("HWC Test Check passed!")
