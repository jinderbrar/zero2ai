
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
        # self.log(f'Patching Linear with activation: {activation}')
        
        # Activation - handle different input types
        if activation is not None:
            # self.log(f'activation of type: {type(activation)}')
            
            # Check if it's a class (not an instance)
            if isinstance(activation, type):
                # self.log(f'activation is a class, checking if subclass of Module')
                if issubclass(activation, Module):
                    # self.log(f'Creating instance of activation class')
                    activation_instance = activation()
                    self.add_module('activation', activation_instance)
                else:
                    # self.log(f'activation is not a Module subclass, treating as function')
                    # Could be a function - store directly
                    self.activation = activation
            
            # If it's already an instance
            elif isinstance(activation, Module):
                # self.log(f'activation is already a Module instance')
                self.add_module('activation', activation)
            
            # If it's a callable (function)
            elif callable(activation):
                # self.log(f'activation is a callable function')
                self.activation = activation
            
            else:
                raise ValueError(f"Invalid activation type: {type(activation)}")
        
        # learning rate
        self.lr = float(kwargs.get('lr', 0.01))

    def _forward(self, x):
        self.log(f'_forward Linear called with input shape {x.shape}')
        self.input = x  # Store input for backward pass
        return np.dot(x, self.weight) + self.bias
    
    def _backward(self, grad_output):
        self.log(f'_backward Linear called with grad_output shape {grad_output.shape}')
        # Compute gradients
        grad_input = np.dot(grad_output, self.weight.T)
        grad_weight = np.dot(self.input.T, grad_output)
        grad_bias = np.sum(grad_output, axis=0)

        # Update parameters (gradient descent)
        self.weight -= self.lr * grad_weight
        self.bias -= self.lr * grad_bias

        return grad_input


class Sequential(Module):

    def __init__(self, *layers, **kwargs):
        super().__init__(**kwargs)
        # self.log(f'adding layer : {layers}')
        for idx, layer in enumerate(layers):
            self.add_module(f'layer_{idx}', layer)

