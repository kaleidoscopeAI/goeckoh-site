"""Raised when pyproject.toml an invalid `build-system.requires`."""

reference = "invalid-pyproject-build-system-requires"

def __init__(self, *, package: str, reason: str) -> None:
    super().__init__(
        message=f"Can not process {escape(package)}",
        context=Text(
            "This package has an invalid `build-system.requires` key in "
            f"pyproject.toml.\n{reason}"
        ),
        note_stmt="This is an issue with the package mentioned above, not pip.",
        hint_stmt=Text("See PEP 518 for the detailed specification."),
    )


