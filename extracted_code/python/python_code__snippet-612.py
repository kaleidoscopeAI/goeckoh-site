    import importlib.machinery
    import importlib.util


def _get_suffixes():
    if imp:
        return [s[0] for s in imp.get_suffixes()]
    else:
        return importlib.machinery.EXTENSION_SUFFIXES


def _load_dynamic(name, path):
    # https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
    if imp:
        return imp.load_dynamic(name, path)
    else:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module


class Mounter(object):

    def __init__(self):
        self.impure_wheels = {}
        self.libs = {}

    def add(self, pathname, extensions):
        self.impure_wheels[pathname] = extensions
        self.libs.update(extensions)

    def remove(self, pathname):
        extensions = self.impure_wheels.pop(pathname)
        for k, v in extensions:
            if k in self.libs:
                del self.libs[k]

    def find_module(self, fullname, path=None):
        if fullname in self.libs:
            result = self
        else:
            result = None
        return result

    def load_module(self, fullname):
        if fullname in sys.modules:
            result = sys.modules[fullname]
        else:
            if fullname not in self.libs:
                raise ImportError('unable to find extension for %s' % fullname)
            result = _load_dynamic(fullname, self.libs[fullname])
            result.__loader__ = self
            parts = fullname.rsplit('.', 1)
            if len(parts) > 1:
                result.__package__ = parts[0]
        return result


