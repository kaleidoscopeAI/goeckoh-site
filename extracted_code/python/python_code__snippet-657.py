class BaseEntryPoint(Protocol):
    @property
    def name(self) -> str:
        raise NotImplementedError()

    @property
    def value(self) -> str:
        raise NotImplementedError()

    @property
    def group(self) -> str:
        raise NotImplementedError()


def _convert_installed_files_path(
    entry: Tuple[str, ...],
    info: Tuple[str, ...],
