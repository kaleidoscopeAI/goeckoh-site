"""Uses the pager installed on the system."""

def _pager(self, content: str) -> Any:  #  pragma: no cover
    return __import__("pydoc").pager(content)

def show(self, content: str) -> None:
    """Use the same pager used by pydoc."""
    self._pager(content)


