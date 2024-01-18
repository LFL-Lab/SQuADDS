class ConfigMaker:

    def __init__(self, component, component_name, data_type):
        self.component = component
        self.component_name = component_name
        self.data_type = data_type
        self.config_name = f"{self.component}_{self.component_name}_{self.data_type}"
        self.metadata = None

    def set_schema(self, ref_file = None, interactive=True):
        """
        # TODO: Implement create_metadata method (both interactive and non-interactive) for required fields
        if interactive:
            self.set_design_fields()
            self.set_sim_options_fields()
            self.set_sim_results_fields()
            self.set_other_fields()
        else:
            self.set_fields(ref_file)
        """
        raise NotImplementedError

    def create_metadata(self, interactive=True):
        raise NotImplementedError

    def submit(self, results):
        package = {
            "data": results,
            "metadata": self.metadata,
        }
        raise NotImplementedError
        
"""
MARKDOWN INFO:

## <a name="creation">Contributing a New Configuration</a>

We may find that we possess a dataset that is not currently included in SQuADDS. In this case, we can add a new configuration to SQuADDS.

But before we do that, we need to make sure that the dataset is in a format that is compatible with the SQuADDS project and is also validated against measurement results. We will go over this process in this section.

### Process Overview

The high level steps for contributing a new configuration are as follows:

1. **Create a new configuration**: Create a new configuration for your dataset.
2. **Create the metadata for the configuration**: Create the metadata for your configuration that contains information on the measured device design and paramaters.
3. **Create the database schema**: Create the database schema for your configuration.
4. **Submit your data**: Submit your data to the SQuADDS database.



"""