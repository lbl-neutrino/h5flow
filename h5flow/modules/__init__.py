from inspect import isclass
import os
import importlib.util
import sys
import logging
from pkgutil import iter_modules


from .h5_flow_dataset_loop_generator import *
from .h5_flow_test_stage import *


def find_class(classname, directory):
    '''
        Search the specified directory for a file containing a python
        implementation with the specified class name

        :param classname: class name to look for

        :param directory: directory to search for ``*.py`` files describing the class

        :returns: ``class`` object of matching desired class (if found), or ``None`` (if not found)
    '''
    path = directory
    for (finder, name, _) in iter_modules([path]):
        if name == 'setup' or name == 'h5flow':
            continue
        try:
            spec = finder.find_spec(name)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            for attribute_name in dir(module):
                attribute = getattr(module, attribute_name)
                if isclass(attribute):
                    if attribute_name == classname:
                        logging.info(f'Using {classname} from {directory}/{name}.py')
                        return attribute
        except Exception as e:
            logging.debug(f'Encountered import error: {e}')
    return None


def get_class(classname, path=None):
    '''
        Look in current directory, ``./h5flow_modules/``, and ``h5flow/modules/``
        for the specified class. Raises a ``RuntimeError`` if class can't be
        found in any of those directories. Optionally, a specific python path can be
        provided and the class will be loaded directly from that module (faster)

        :param classname: class name to search for

        :param path: python path to module that class can be accessed from

        :returns: ``class`` object of desired class

    '''
    search_only_path = True
    if path is None:
        path = './'
        search_only_path = False

    if not search_only_path:
        # first search in local directory
        found_class = find_class(classname, path)
        
        if found_class is None:
            found_class = find_class(classname, 'h5flow_modules/')

        if found_class is None:
            for d in ('h5flow_modules/','./'):
                # then recurse into subdirectories
                for parent, dirs, files in os.walk(d, followlinks=True):
                    for directory in dirs:
                        found_class = find_class(classname, os.path.join(parent, directory))
                        if found_class is not None:
                            break
                    if found_class is not None:
                        break

            if found_class is None:
                # then search in source
                found_class = find_class(classname, os.path.dirname(__file__))
    else:
        _tmp = __import__(path, globals(), locals(), [classname], 0)
        found_class = getattr(_tmp, classname)

    if found_class is None:
        raise RuntimeError(f'no matching class {classname} found!')

    return found_class
