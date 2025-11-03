

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
    

class LeakyReLU(Module):

    def __init__(self, alpha=0.01, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
    
    def forward(self, x):
        self.x = x
        return np.where(x>0, x, self.alpha * x)
    
    def backward(self, grad_output):
        return grad_output * np.where(self.x>0, 1, self.alpha)


class Sigmoid(Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x):
        self.x = x  # Store input for backward pass
        return 1 / (1 + np.exp(-x))

    def backward(self, grad_output):
        sigmoid_x = 1 / (1 + np.exp(-self.x))
        return grad_output * sigmoid_x * (1 - sigmoid_x)    


class Softmax(Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def forward(self, z):
        # numerical stability
        z_shifted = z - np.max(z, axis=-1, keepdims=True)
        exp_z = np.exp(z_shifted)
        # output is required in gradient calculation
        self.out = exp_z / np.sum(exp_z, axis=-1, keepdims=True)
        return self.out
    
    def backward(self, grad_output):
        
        s = self.out.reshape(-1, 1)
        J = np.diagflat(s) - np.dot(s, s.T)
        grad_input = np.dot(J, grad_output)
        return grad_input



    



