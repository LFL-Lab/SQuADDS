import os

__version__ = '0.1.7'
__license__ = "MIT License"
__copyright__ = 'Sadman Ahmed Shanto, Eli Levenson-Falk 2023'
__author__ = 'Sadman Ahmed Shanto, Eli Levenson-Falk'
__status__ = "Alpha"
__repo_path__ = os.path.dirname(os.path.abspath(__file__))

from squadds.core.db import SQuADDS_DB
from squadds.core.analysis import Analyzer
from squadds.simulations.ansys_simulator import AnsysSimulator
