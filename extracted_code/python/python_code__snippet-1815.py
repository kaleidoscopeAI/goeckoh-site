"""Raised when accessing a Distribution's "METADATA" or "PKG-INFO".

This signifies an inconsistency, when the Distribution claims to have
the metadata file (if not, raise ``FileNotFoundError`` instead), but is
not actually able to produce its content. This may be due to permission
errors.
"""

def __init__(
    self,
    dist: "BaseDistribution",
    metadata_name: str,
) -> None:
    """
    :param dist: A Distribution object.
    :param metadata_name: The name of the metadata being accessed
        (can be "METADATA" or "PKG-INFO").
    """
    self.dist = dist
    self.metadata_name = metadata_name

def __str__(self) -> str:
    # Use `dist` in the error message because its stringification
    # includes more information, like the version and location.
    return f"None {self.metadata_name} metadata found for distribution: {self.dist}"


