import os

__version__ = "0.5.1"
__license__ = "MIT License"
__copyright__ = "Sadman Ahmed Shanto, Eli Levenson-Falk 2023"
__author__ = "Sadman Ahmed Shanto"
__status__ = "Alpha"
__repo_path__ = os.path.dirname(os.path.abspath(__file__))


# Lazy imports: AnsysSimulator requires Qt/qiskit_metal (needs a display); Analyzer and SQuADDS_DB pull in the
# scientific stack, which squadds.ml does not need.
def __getattr__(name):
    if name == "Analyzer":
        from squadds.core.analysis import Analyzer

        return Analyzer
    if name == "SQuADDS_DB":
        from squadds.core.db import SQuADDS_DB

        return SQuADDS_DB
    if name == "AnsysSimulator":
        from squadds.simulations.ansys_simulator import AnsysSimulator

        return AnsysSimulator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
