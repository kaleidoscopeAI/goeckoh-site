"""
Encapsulates some of the preferences for filtering and sorting
InstallationCandidate objects.
"""

def __init__(
    self,
    prefer_binary: bool = False,
    allow_all_prereleases: bool = False,
) -> None:
    """
    :param allow_all_prereleases: Whether to allow all pre-releases.
    """
    self.allow_all_prereleases = allow_all_prereleases
    self.prefer_binary = prefer_binary


