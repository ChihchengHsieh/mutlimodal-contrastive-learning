from .dataset import *
from .model import *
from .training import *

class ConfigArgs():
    def __init__(self, training, dataset, model):
        self.training = training
        self.dataset = dataset
        self.model = model

    def __str__(self):
        return str(self.training) + "\n" + str(self.dataset) + "\n" + str(self.model)