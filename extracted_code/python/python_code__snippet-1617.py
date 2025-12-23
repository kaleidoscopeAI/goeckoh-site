"""A column to insert an arbitrary column.

Args:
    renderable (RenderableType, optional): Any renderable. Defaults to empty string.
"""

def __init__(
    self, renderable: RenderableType = "", *, table_column: Optional[Column] = None
):
    self.renderable = renderable
    super().__init__(table_column=table_column)

def render(self, task: "Task") -> RenderableType:
    return self.renderable


