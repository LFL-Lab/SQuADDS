import os
from abc import ABC, abstractmethod
from datetime import datetime


class AbstractUploadData(ABC):

    def __init__(self, config_name):
        self.config_name = config_name
        self._set_contributor_info()
    
    @abstractmethod
    def _validate_config_name(self):
        pass

    @abstractmethod
    def get_config_schema(self):
        pass

    @abstractmethod
    def show_config_schema(self):
        pass

    @abstractmethod
    def _supported_config_names(self):
        pass

    @abstractmethod
    def show(self):
        pass

    @abstractmethod
    def add_sim_result(self, result_name, result_value, unit):
        pass

    @abstractmethod
    def add_sim_setup(self, sim_setup):
        pass

    @abstractmethod
    def add_design(self, design):
        pass

    @abstractmethod
    def to_dict(self):
        pass

    @abstractmethod
    def clear(self):
        pass

    @abstractmethod
    def add_notes(self, notes={}):
        pass

    @abstractmethod
    def _validate_structure(self):
        pass

    @abstractmethod
    def _validate_types(self):
        pass

    @abstractmethod
    def _validate_content(self):
        pass

    @abstractmethod
    def validate(self):
        pass

    @abstractmethod
    def create_PR(self):
        pass

    @abstractmethod
    def submit(self):
        pass

    @abstractmethod
    def _set_contributor_info(self):
        self.contributor = {
            "group": os.getenv('GROUP_NAME'),
            "PI": os.getenv('PI_NAME'),
            "institution": os.getenv('INSTITUTION'),
            "uploader": os.getenv('USER_NAME'),
            "misc": os.getenv('CONTRIB_MISC'),
            "date_created": datetime.now().strftime("%Y-%m-%d %H%M%S")
        }
