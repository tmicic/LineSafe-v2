import torch
import torch.nn as nn
import torchvision.transforms as transforms
from numpy.random import choice


class ToMultiChannel:

    def __init__(self, number_of_channels=3):
        
        self.number_of_channels = number_of_channels

    def __call__(self, x):

        return torch.cat([x]*self.number_of_channels, dim=0)
