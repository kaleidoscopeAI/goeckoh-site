class InstallationResult:
    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"InstallationResult(name={self.name!r})"


def _validate_requirements(
    requirements: List[InstallRequirement],
