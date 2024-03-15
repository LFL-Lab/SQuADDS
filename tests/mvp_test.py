import matplotlib

matplotlib.use('Agg')  # Set the backend to Agg

from squadds import Analyzer, SQuADDS_DB
from squadds.interpolations.physics import ScalingInterpolator

try:
    db = SQuADDS_DB()
    db.view_datasets()
    db.get_configs()
    db.get_dataset_info(component="qubit", component_name="TransmonCross", data_type="cap_matrix")
    db.get_dataset_info(component="cavity_claw", component_name="RouteMeander", data_type="eigenmode")
    db.see_dataset(component="qubit", component_name="TransmonCross", data_type="cap_matrix")
    db.view_all_contributors()

    db.unselect_all()
    db.select_system(["qubit","cavity_claw"])
    db.select_qubit("TransmonCross")
    db.select_cavity_claw("RouteMeander")
    db.select_coupler("CLT")
    db.show_selections()
    merged_df = db.create_system_df()
    print(merged_df)

    analyzer = Analyzer(db)

    target_params = {
                    "qubit_frequency_GHz": 4,
                    "cavity_frequency_GHz": 6.2,
                    "kappa_kHz": 120,
                    "resonator_type":"quarter",
                    "anharmonicity_MHz": -200,
                    "g_MHz": 70}

    results = analyzer.find_closest(target_params=target_params,
                                           num_top=3,
                                           metric="Euclidean",
                                           display=True)
    print(results)

    interpolator = ScalingInterpolator(analyzer, target_params)

    design_df = interpolator.get_design()

    print(design_df)

except Exception as e:
    print(f"An error occurred: {str(e)}")
except Exception as e:
    print(f"An error occurred: {str(e)}")
