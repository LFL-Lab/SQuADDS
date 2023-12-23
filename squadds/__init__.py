import os

__version__ = '0.1.6'
__license__ = "MIT License"
__copyright__ = 'Sadman Ahmed Shanto, Eli Levenson-Falk 2023'
__author__ = 'Sadman Ahmed Shanto, Eli Levenson-Falk'
__status__ = "Alpha"
__repo_path__ = os.path.dirname(os.path.abspath(__file__))
__library_path__ = os.path.join(__repo_path__, "library")


from .core.db import SQuADDS_DB
from .core.analysis import Analyzer
