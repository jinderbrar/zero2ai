
from typing import Optional, Union, Callable, Type

import numpy as np
from ..core import Module


class Linear(Module):

    def __init__(self, fan_in, fan_out, **kwargs):
        super().__init__(**kwargs)
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.weight = np.random.randn(fan_in, fan_out) * np.sqrt(2. / fan_in)
        self.bias = np.zeros(fan_out)

        self.patch(**kwargs)

    def patch(self, **kwargs):
        activation = kwargs.get('activation', None)
        self.log(f'Patching Linear with activation: {activation}')
        if activation is not None:
            self.log(f'activation of type: {type(activation)}')
            if issubclass(activation, Module):
                self.add_module('activation', activation())

    def _forward(self, x):
        self.log(f'_forward Linear called with input shape {x.shape}')
        return np.dot(x, self.weight) + self.bias
    
    def backward(self, grad_output):
        # Placeholder for backward pass
        # In a real implementation, compute gradients w.r.t. weights, bias, and input
        pass


class Sequential(Module):

    def __init__(self, *layers):
        super().__init__()
        for idx, layer in enumerate(layers):
            self.add_module(f'layer_{idx}', layer)

