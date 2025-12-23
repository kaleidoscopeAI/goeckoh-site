"""Base class for a widget to use in progress display."""

max_refresh: Optional[float] = None

def __init__(self, table_column: Optional[Column] = None) -> None:
    self._table_column = table_column
    self._renderable_cache: Dict[TaskID, Tuple[float, RenderableType]] = {}
    self._update_time: Optional[float] = None

def get_table_column(self) -> Column:
    """Get a table column, used to build tasks table."""
    return self._table_column or Column()

def __call__(self, task: "Task") -> RenderableType:
    """Called by the Progress object to return a renderable for the given task.

    Args:
        task (Task): An object containing information regarding the task.

    Returns:
        RenderableType: Anything renderable (including str).
    """
    current_time = task.get_time()
    if self.max_refresh is not None and not task.completed:
        try:
            timestamp, renderable = self._renderable_cache[task.id]
        except KeyError:
            pass
        else:
            if timestamp + self.max_refresh > current_time:
                return renderable

    renderable = self.render(task)
    self._renderable_cache[task.id] = (current_time, renderable)
    return renderable

@abstractmethod
def render(self, task: "Task") -> RenderableType:
    """Should return a renderable object."""


