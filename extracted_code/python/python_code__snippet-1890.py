def __init__(self, entry_point: str) -> None:
    super().__init__(
        f"Invalid script entry point: {entry_point} - A callable "
        "suffix is required. Cf https://packaging.python.org/"
        "specifications/entry-points/#use-for-scripts for more "
        "information."
    )


