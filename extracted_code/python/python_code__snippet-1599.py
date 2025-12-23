"""Base class for a pager."""

@abstractmethod
def show(self, content: str) -> None:
    """Show content in pager.

    Args:
        content (str): Content to be displayed.
    """


