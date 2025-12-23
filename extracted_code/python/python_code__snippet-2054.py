def __init__(self, path: str) -> None:
    self.path = path
    self.setup = False
    scheme = get_scheme("", prefix=path)
    self.bin_dir = scheme.scripts
    self.lib_dirs = _dedup(scheme.purelib, scheme.platlib)


