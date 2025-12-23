"""
A meta path importer to import six.moves and its submodules.

This class implements a PEP302 finder and loader. It should be compatible
with Python 2.5 and all existing versions of Python3
"""

def __init__(self, six_module_name):
    self.name = six_module_name
    self.known_modules = {}

def _add_module(self, mod, *fullnames):
    for fullname in fullnames:
        self.known_modules[self.name + "." + fullname] = mod

def _get_module(self, fullname):
    return self.known_modules[self.name + "." + fullname]

def find_module(self, fullname, path=None):
    if fullname in self.known_modules:
        return self
    return None

def find_spec(self, fullname, path, target=None):
    if fullname in self.known_modules:
        return spec_from_loader(fullname, self)
    return None

def __get_module(self, fullname):
    try:
        return self.known_modules[fullname]
    except KeyError:
        raise ImportError("This loader does not know module " + fullname)

def load_module(self, fullname):
    try:
        # in case of a reload
        return sys.modules[fullname]
    except KeyError:
        pass
    mod = self.__get_module(fullname)
    if isinstance(mod, MovedModule):
        mod = mod._resolve()
    else:
        mod.__loader__ = self
    sys.modules[fullname] = mod
    return mod

def is_package(self, fullname):
    """
    Return true, if the named module is a package.

    We need this method to get correct spec objects with
    Python 3.4 (see PEP451)
    """
    return hasattr(self.__get_module(fullname), "__path__")

def get_code(self, fullname):
    """Return None

    Required, if is_package is implemented"""
    self.__get_module(fullname)  # eventually raises ImportError
    return None
get_source = get_code  # same as get_code

def create_module(self, spec):
    return self.load_module(spec.name)

def exec_module(self, module):
    pass

