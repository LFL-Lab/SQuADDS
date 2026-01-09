import os

__version__ = "0.4.0"
__license__ = "MIT License"
__copyright__ = "Sadman Ahmed Shanto, Eli Levenson-Falk 2023"
__author__ = "Sadman Ahmed Shanto"
__status__ = "Alpha"
__repo_path__ = os.path.dirname(os.path.abspath(__file__))

from squadds.core.analysis import Analyzer
from squadds.core.db import SQuADDS_DB
from squadds.simulations.ansys_simulator import AnsysSimulator

__all__ = [
    "Analyzer",
    "SQuADDS_DB",
    "AnsysSimulator",
    "__version__",
    "__license__",
    "__copyright__",
    "__author__",
    "__status__",
    "__repo_path__",
]
