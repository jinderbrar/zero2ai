"""
utility functions for loss calculations
"""

import numpy as np


"""MSE loss and grad"""


class mse:
    
    @staticmethod
    def loss(ypred, ytrue):
        return np.mean((ypred - ytrue)**2)
    
    @staticmethod
    def grad(ypred, ytrue):
        n = ypred.size
        return (2.0 / n) * (ypred - ytrue)
    
    def __call__(self, ypred, ytrue):
        return self.loss(ypred, ytrue)
