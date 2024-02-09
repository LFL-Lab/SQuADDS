from squadds.core.utils import set_huggingface_api_key

set_huggingface_api_key()

from datasets import get_dataset_config_names
from datasets import load_dataset
from squadds import SQuADDS_DB
from squadds.interpolations.physics import ScalingInterpolator
from squadds import Analyzer
from squadds import AnsysSimulator
import numpy as np

def main():
    db = SQuADDS_DB()
    configs = get_dataset_config_names("SQuADDS/SQuADDS_DB")
    db.select_system(['qubit','cavity_claw'])
    db.select_qubit('TransmonCross')
    db.select_cavity_claw('RouteMeander')
    db.select_coupler('CLT')
    df = db.create_system_df()

    analyzer = Analyzer(db)
    analyzer.target_param_keys()
    target_params = {
                        "qubit_frequency_GHz": 5.523,
                        "cavity_frequency_GHz": 7.815,
                        "kappa_kHz": 400,
                        "resonator_type":"quarter",
                        "anharmonicity_MHz": -200,
                        "g_MHz": 30}

    results = analyzer.find_closest(target_params=target_params,
                                            num_top=3,
                                            metric="Euclidean",
                                            display=True)
    interpolator = ScalingInterpolator(analyzer, target_params)
    design_df = interpolator.get_design()
    best_device = results.iloc[0]
    simulator = AnsysSimulator(analyzer, best_device)
    design_options = best_device['design_options']

    design_options_dict_list = []
    simulated_params_list = []   
    sweep_list = list(np.arange(0,20.1,1))
    best_device['design_options_qubit']['connection_pads']['readout']['ground_spacing'] = sweep_list
    # design_options['ground_spacing'] = sweep_list
    # design_df
    # ansys_results = simulator.simulate(best_device)
    sweep_results = simulator.sweep_qubit_cavity(best_device)
    # design_options = ansys_results['design']['design_options']
    # sim_results = ansys_results['sim_results']
    # design_options_dict_list.append(design_options)
    # simulated_params_list.append(sim_results)
    print(sweep_results)

if __name__ == "__main__":
    main()

