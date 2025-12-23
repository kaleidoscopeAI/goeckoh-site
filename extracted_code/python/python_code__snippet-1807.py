"""A SimpleWheelCache that creates it's own temporary cache directory"""

def __init__(self) -> None:
    self._temp_dir = TempDirectory(
        kind=tempdir_kinds.EPHEM_WHEEL_CACHE,
        globally_managed=True,
    )

    super().__init__(self._temp_dir.path)


