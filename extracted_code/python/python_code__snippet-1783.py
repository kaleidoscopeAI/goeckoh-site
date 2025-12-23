"""Provides access to package resources in the filesystem"""

def _has(self, path):
    return os.path.exists(path)

def _isdir(self, path):
    return os.path.isdir(path)

def _listdir(self, path):
    return os.listdir(path)

def get_resource_stream(self, manager, resource_name):
    return open(self._fn(self.module_path, resource_name), 'rb')

def _get(self, path):
    with open(path, 'rb') as stream:
        return stream.read()

@classmethod
def _register(cls):
    loader_names = (
        'SourceFileLoader',
        'SourcelessFileLoader',
    )
    for name in loader_names:
        loader_cls = getattr(importlib_machinery, name, type(None))
        register_loader_type(loader_cls, cls)


