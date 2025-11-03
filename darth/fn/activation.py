

"""
Generic activation fn and class that works on any input shape
"""

from typing import Union
import numpy as np

from ..core import Module


class ReLU(Module):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x):
        self.x = x  # Store input for backward pass
        return np.maximum(0, x)

    def backward(self, grad_output):
        return grad_output * (self.x > 0)


class Sigmoid(Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x):
        self.x = x  # Store input for backward pass
        return 1 / (1 + np.exp(-x))

    def backward(self, grad_output):
        sigmoid_x = 1 / (1 + np.exp(-self.x))
        return grad_output * sigmoid_x * (1 - sigmoid_x)    




