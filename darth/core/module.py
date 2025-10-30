"""
Base class that every other module will inherit from
providing basic functionalities
"""

import numpy as np
from typing import Any
from collections import OrderedDict
import inspect, types

class Module:

    def __init__(self):
        self._name = self.__class__.__name__
        self._submodules = OrderedDict()
        self._params = OrderedDict()
    
    @property
    def name(self) -> str:
        return self._name
    
    def add_module(self, name: str, module: 'Module') -> None:
        self._submodules[name] = module
    
    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, Module):
            self._submodules[name] = value
        else:
            if name in inspect.signature(self.__init__).parameters:
                self._params[name] = value
            object.__setattr__(self, name, value)

    def __repr__(self,d=0) -> str:
        """
        Simple recursive repr for easy debugging
        Shows submodules and their parameters
        """
        indent = '  ' * (abs(d)+1)
        repr = f"{self._name}(\n"
        for name, module in self._submodules.items():
            repr += f"{indent}{name} = {module.__repr__(d=d+1)},\n"
        for name, value in self.__dict__.items():
            if inspect.isfunction(value) or inspect.ismethod(value):
                repr += f"{indent}{name} = {value.__name__},\n"
            elif name in self._params:
                repr += f"{indent}{name} = {value},\n"
        
        repr += f"{indent})"
        return repr
    
    def _forward(self, x, *args, **kwargs):
        return x

    def forward(self, *args, **kwargs):
        """
        Forward pass through the module and its submodules.
        """
        x = self._forward(*args, **kwargs)
        for module in self._submodules.values():
            x = module.forward(x)
        return x
