def select_backend() -> Backend:
    if _should_use_importlib_metadata():
        from . import importlib

        return cast(Backend, importlib)
    from . import pkg_resources

    return cast(Backend, pkg_resources)


def get_default_environment() -> BaseEnvironment:
    """Get the default representation for the current environment.

    This returns an Environment instance from the chosen backend. The default
    Environment instance should be built from ``sys.path`` and may use caching
    to share instance state accorss calls.
    """
    return select_backend().Environment.default()


def get_environment(paths: Optional[List[str]]) -> BaseEnvironment:
    """Get a representation of the environment specified by ``paths``.

    This returns an Environment instance from the chosen backend based on the
    given import paths. The backend must build a fresh instance representing
    the state of installed distributions when this function is called.
    """
    return select_backend().Environment.from_paths(paths)


def get_directory_distribution(directory: str) -> BaseDistribution:
    """Get the distribution metadata representation in the specified directory.

    This returns a Distribution instance from the chosen backend based on
    the given on-disk ``.dist-info`` directory.
    """
    return select_backend().Distribution.from_directory(directory)


def get_wheel_distribution(wheel: Wheel, canonical_name: str) -> BaseDistribution:
    """Get the representation of the specified wheel's distribution metadata.

    This returns a Distribution instance from the chosen backend based on
    the given wheel's ``.dist-info`` directory.

    :param canonical_name: Normalized project name of the given wheel.
    """
    return select_backend().Distribution.from_wheel(wheel, canonical_name)


def get_metadata_distribution(
    metadata_contents: bytes,
    filename: str,
    canonical_name: str,
