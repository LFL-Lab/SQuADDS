"""
Helper methods to create config files
"""

from datasets import BuilderConfig


class SQuADDS_DB_Config(BuilderConfig):
    """BuilderConfig for SQuADDS_DB."""

    def __init__(self, circuit_element=None, element_name=None, result_type=None, **kwargs):
        super().__init__(**kwargs)
        self.circuit_element = circuit_element
        self.element_name = element_name
        self.result_type = result_type
