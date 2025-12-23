"""Provider based on a virtual filesystem"""

def __init__(self, module):
    super().__init__(module)
    self._setup_prefix()

def _setup_prefix(self):
    # Assume that metadata may be nested inside a "basket"
    # of multiple eggs and use module_path instead of .archive.
    eggs = filter(_is_egg_path, _parents(self.module_path))
    egg = next(eggs, None)
    egg and self._set_egg(egg)

def _set_egg(self, path):
    self.egg_name = os.path.basename(path)
    self.egg_info = os.path.join(path, 'EGG-INFO')
    self.egg_root = path


