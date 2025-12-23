"""Provides hooks in to the render process."""

@abstractmethod
def process_renderables(
    self, renderables: List[ConsoleRenderable]
) -> List[ConsoleRenderable]:
    """Called with a list of objects to render.

    This method can return a new list of renderables, or modify and return the same list.

    Args:
        renderables (List[ConsoleRenderable]): A number of renderable objects.

    Returns:
        List[ConsoleRenderable]: A replacement list of renderables.
    """


