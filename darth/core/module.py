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
        self._name = self.__class__.__name__
        self._submodules = OrderedDict()
        self._params = OrderedDict()
        self.activation = None  # Default activation is None

    @property
    def name(self) -> str:
        return self._name

    def add_module(self, name: str, module) -> None:
        self.log(f'Adding submodule: {name} of type {module.__class__.__name__}')
        module_obj = module() if isinstance(module, type) else module
        self._submodules[name] = module
    
    def __setattr__(self, name: str, value: Any) -> None:
        # print(f' Setting attribute {name} to {value}')
        if isinstance(value, Module):
            # print(f'  Setting submodule {name} to {value}')
            self._submodules[name] = value
        else:
            if name in inspect.signature(self.__init__).parameters:
                self._params[name] = value
            # print(f'  Setting normal attribute {name} to {value}')
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

        # print(f'representation so far:\n{repr} {len(repr)} chars')

        return repr

    
    def __str__(self) -> str:
        return self._name

    def log(self, message: str) -> None:
        print(f"[{self._name}] {message}")

class Module(DownToEarthModule):
    """Base class for all neural network modules."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    
    def _forward(self, x, *args, **kwargs):
        self.log('_forward called - should be overridden in subclass')
        return x

    def forward(self, *args, **kwargs):
        """
        Forward pass through the module and its submodules.
        """
        self.log('forward called')
        x = self._forward(*args, **kwargs)
        for module in self._submodules.values():
            x = module.forward(x)
        self.x = x  # Store output for potential backward pass
        return self.apply_activation(x)
    
    def apply_activation(self, x):
        """
        Apply activation function if defined.
        """
        self.log(f'apply_activation called with activation: {self.activation}')
        if self.activation is not None:
            return self.activation.forward(x)
        return x
