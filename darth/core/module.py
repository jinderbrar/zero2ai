"""
Base class that every other module will inherit from
providing basic functionalities
"""

import numpy as np
from typing import Any
from collections import OrderedDict
import inspect, types

class DownToEarthModule:

    """Base class for all modules, providing basic functionalities."""
    def __init__(self, **kwargs):
        self._submodules = OrderedDict()
        self._params = OrderedDict()
        self.activation = None  # Default activation is None
        self._name = kwargs.get('name', self.__class__.__name__)

    @property
    def name(self) -> str:
        return self._name

    def add_module(self, name: str, module) -> None:
        # self.log(f'Adding submodule: {name} of type {module.name}')
        module_obj = module(name=name) if isinstance(module, type) else module
        self._submodules[name] = module_obj
    
    def __setattr__(self, name: str, value: Any) -> None:
        
        # Check if it's a Module (but only after _submodules is initialized)
        if hasattr(self, '_submodules') and isinstance(value, DownToEarthModule):
            self._submodules[name] = value
            object.__setattr__(self, name, value)
        else:
            # Check if it's a parameter from __init__
            if hasattr(self, '_params') and name in inspect.signature(self.__init__).parameters:
                self._params[name] = value
            object.__setattr__(self, name, value)

    def __repr__(self,d=0) -> str:
        """
        Simple recursive repr for easy debugging
        Shows submodules and their parameters
        """
        indent = '  ' * (abs(d)+1)

        _initial = f"{self.name}("
        repr = _initial
        # check params
        for name, value in self.__dict__.items():
            if inspect.isfunction(value) or inspect.ismethod(value):
                repr += f"\n{indent}{name} = {value.__name__},"
            elif name in self._params:
                repr += f"\n{indent}{name} = {value},"
        
        # check submodules
        for name, module in self._submodules.items():
            repr += f"\n{indent}{name} = {module.__repr__(d=d+1)},"
        
        if repr == _initial:
            repr += ")"
        else:
            repr += f"\n{'  '*d})"

        return repr

    
    def __str__(self) -> str:
        return self.__repr__()

    def log(self, message: str) -> None:
        if getattr(self, 'verbose', False):
            print(f"[{self._name}] {message}")

class Module(DownToEarthModule):
    """Base class for all neural network modules."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.verbose = kwargs.get('verbose', False)

    
    def _forward(self, x, *args, **kwargs):
        self.log('_forward called - should be overridden in subclass')
        return x

    def forward(self, *args, **kwargs):
        """
        Forward pass through the module and its submodules.
        """
        self.log('forward called')
        x = self._forward(*args, **kwargs)

        # apply submodules in order (excluding activation)
        submodules = [(name, mod) for name, mod in self._submodules.items() if name != 'activation']
        for name, module in submodules:
            # self.log(f'forwarding through submodule: {module.name}')
            x = module.forward(x)
        self.x = x  # Store output for potential backward pass
        return self.apply_activation(x)

    def _backward(self, grad_output, *args, **kwargs):
        self.log('_backward called - should be overridden in subclass')
        return grad_output
    
    def backward(self, grad_output, *args, **kwargs):
        """
        Backward pass through the module and its submodules.
        Processes in REVERSE order compared to forward pass.
        """
        self.log('backward called')
        
        # First, backprop through activation if it exists
        grad_output = self.apply_activation_backward(grad_output)
        
        # Then through the layer's own backward
        grad_output = self._backward(grad_output, *args, **kwargs)
        
        # Finally through submodules in REVERSE order (excluding activation)
        submodules = [(name, mod) for name, mod in self._submodules.items() if name != 'activation']
        for name, module in reversed(submodules):
            # self.log(f'Backward on: {module.name}, with shape: {grad_output.shape}')
            grad_output = module.backward(grad_output, *args, **kwargs)
        
        return grad_output
    
    def apply_activation_backward(self, grad_output):
        """
        Apply activation backward if defined.
        """
        activation = self.activation or self._submodules.get('activation', None)
        # self.log(f'activation being applied backward with : {activation}')
        if activation is not None:
            return activation.backward(grad_output)
        return grad_output
    
    def apply_activation(self, x):
        """
        Apply activation function if defined.
        """
        activation = self.activation or self._submodules.get('activation', None)
        # self.log(f'activation being applied with : {activation}')
        if activation is not None:
            return activation.forward(x)
        return x
