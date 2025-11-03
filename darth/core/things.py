import types
from collections import defaultdict

# global reg
GLOBAL_REG = defaultdict(None)

def register_module(module):
    """Decorator to register a module in the global registry."""
    if not isinstance(module, types.ModuleType):
        raise ValueError("Can only register modules. Got: {}".format(type(module)))
    
    module_name = module.__name__
    print(f"Registering module: {module_name}")
    for name, obj in vars(module).items():
        # only register classes from this module
        print(f"obj.__module__: {obj.__module__}, module_name: {module_name}")
        if isinstance(obj, type) and obj.__module__ == module_name:
            GLOBAL_REG[name] = obj
            print(f"  Registered class: {name}")
