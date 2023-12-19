# Dataset upload idea:

- [ ] upload data for LOM via script
- [ ] upload data for eigenmodal via script

- dataset format:

    {"design",
    "simulation_hyperparameters",
    "simulation_results",
    "notes"
    }

- metadata format:

    {
     "measured_design_code" or "measured_design_params",
     "measured_hamiltonian_params",
     "notes"
    },

    {"verified_design",
    "verified_simulation_hyperparameters",
    "verified_simulation_results",
    "notes"
    }
 
- uid of dataset stored in `DatasetInfo, DatasetCard` and connected somehow via something

- preprocessing of datasets