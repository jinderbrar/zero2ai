
import numpy as np
from ..core import Module
from .. import fn

class Linear(Module):

    def __init__(self, fan_in, fan_out, actFn=fn.relu):
        super().__init__()
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.actFn = actFn
        self.weight = np.random.randn(fan_in, fan_out) * np.sqrt(2. / fan_in)
        self.bias = np.zeros(fan_out)

    def forward(self, x):
        self.x = x  # Store input for potential backward pass
        _x = np.dot(x, self.weight) + self.bias
        if self.actFn is not None:
            _x = self.actFn(_x)
        return _x
    
    def backward(self, grad_output):
        # Placeholder for backward pass
        # In a real implementation, compute gradients w.r.t. weights, bias, and input
        pass


class Sequential(Module):

    def __init__(self, *layers):
        super().__init__()
        for idx, layer in enumerate(layers):
            self.add_module(f'layer_{idx}', layer)

    def forward(self, x):
        self.x = x  # Store input for potential backward pass
        for layer in self._submodules.values():
            x = layer.forward(x)
        return x
