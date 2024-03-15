from abc import ABC, abstractmethod

class Simulator(ABC):
    
    @abstractmethod
    def get_design_screenshot():
        pass