reference = "metadata-generation-failed"

def __init__(
    self,
    *,
    package_details: str,
) -> None:
    super(InstallationSubprocessError, self).__init__(
        message="Encountered error while generating package metadata.",
        context=escape(package_details),
        hint_stmt="See above for details.",
        note_stmt="This is an issue with the package mentioned above, not pip.",
    )

def __str__(self) -> str:
    return "metadata generation failed"


