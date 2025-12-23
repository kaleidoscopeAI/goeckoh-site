"""Raised when pyproject.toml has `build-system`, but no `build-system.requires`."""

reference = "missing-pyproject-build-system-requires"

def __init__(self, *, package: str) -> None:
    super().__init__(
        message=f"Can not process {escape(package)}",
        context=Text(
            "This package has an invalid pyproject.toml file.\n"
            "The [build-system] table is missing the mandatory `requires` key."
        ),
        note_stmt="This is an issue with the package mentioned above, not pip.",
        hint_stmt=Text("See PEP 518 for the detailed specification."),
    )


