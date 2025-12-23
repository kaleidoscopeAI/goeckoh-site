"""Renders completed count/total, e.g. '  10/1000'.

Best for bounded tasks with int quantities.

Space pads the completed count so that progress length does not change as task progresses
past powers of 10.

Args:
    separator (str, optional): Text to separate completed and total values. Defaults to "/".
"""

def __init__(self, separator: str = "/", table_column: Optional[Column] = None):
    self.separator = separator
    super().__init__(table_column=table_column)

def render(self, task: "Task") -> Text:
    """Show completed/total."""
    completed = int(task.completed)
    total = int(task.total) if task.total is not None else "?"
    total_width = len(str(total))
    return Text(
        f"{completed:{total_width}d}{self.separator}{total}",
        style="progress.download",
    )


