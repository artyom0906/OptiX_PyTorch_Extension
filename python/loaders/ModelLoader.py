from abc import ABC, abstractmethod

from python.loaders.Model import Model


class ModelLoader(ABC):
    def __init__(self):
        self.model = None

    @abstractmethod
    def load(self)->Model:
        """
        Load the model data (geometry and textures) from the file.
        """
        pass