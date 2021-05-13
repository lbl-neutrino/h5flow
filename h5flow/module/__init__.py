from inspect import isclass
from pkgutil import iter_modules
import os
from importlib import import_module

def get_class(classname):
    path = os.path.dirname('./h5flow_modules/')
    for (_, module_name, _) in iter_modules([path]):
        module = import_module(f"h5flow_modules.{module_name}")
        for attribute_name in dir(module):
            attribute = getattr(module, attribute_name)

            if isclass(attribute) and attribute_name == classname:
                return attribute
    raise RuntimeError(f'no matching class {classname} found in {path}')
