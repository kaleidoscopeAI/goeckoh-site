"""Base class for a splitter."""

name: str = ""

@abstractmethod
def get_tree_icon(self) -> str:
    """Get the icon (emoji) used in layout.tree"""

@abstractmethod
def divide(
    self, children: Sequence["Layout"], region: Region
) -> Iterable[Tuple["Layout", Region]]:
    """Divide a region amongst several child layouts.

    Args:
        children (Sequence(Layout)): A number of child layouts.
        region (Region): A rectangular region to divide.
    """


