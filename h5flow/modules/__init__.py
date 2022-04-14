from inspect import isclass
import os
import importlib.util
import sys
import logging
from pkgutil import iter_modules


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


def get_class(classname):
    '''
        Look in current directory, ``./h5flow_modules/``, and ``h5flow/modules/``
        for the specified class. Raises a ``RuntimeError`` if class can't be
        found in any of those directories.

        :param classname: class name to search for

        :returns: ``class`` object of desired class

    '''

    # first search in local directory
    found_class = find_class(classname, './')

    if found_class is None:
        # then recurse into subdirectories
        if found_class is None:
            for parent, dirs, files in os.walk('./', followlinks=True):
                for directory in dirs:
                    found_class = find_class(classname, os.path.join(parent, directory))
                    if found_class is not None:
                        break
                if found_class is not None:
                    break

        if found_class is None:
            # then search in source
            found_class = find_class(classname, os.path.dirname(__file__))

    if found_class is None:
        raise RuntimeError(f'no matching class {classname} found!')

    return found_class
