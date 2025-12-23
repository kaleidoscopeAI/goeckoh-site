"""A no-op drop-in replacement for BuildEnvironment"""

def __init__(self) -> None:
    pass

def __enter__(self) -> None:
    pass

def __exit__(
    self,
    exc_type: Optional[Type[BaseException]],
    exc_val: Optional[BaseException],
    exc_tb: Optional[TracebackType],
) -> None:
    pass

def cleanup(self) -> None:
    pass

def install_requirements(
    self,
    finder: "PackageFinder",
    requirements: Iterable[str],
    prefix_as_string: str,
    *,
    kind: str,
) -> None:
    raise NotImplementedError()


