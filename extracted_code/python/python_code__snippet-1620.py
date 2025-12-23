"""Renders a visual progress bar.

Args:
    bar_width (Optional[int], optional): Width of bar or None for full width. Defaults to 40.
    style (StyleType, optional): Style for the bar background. Defaults to "bar.back".
    complete_style (StyleType, optional): Style for the completed bar. Defaults to "bar.complete".
    finished_style (StyleType, optional): Style for a finished bar. Defaults to "bar.finished".
    pulse_style (StyleType, optional): Style for pulsing bars. Defaults to "bar.pulse".
"""

def __init__(
    self,
    bar_width: Optional[int] = 40,
    style: StyleType = "bar.back",
    complete_style: StyleType = "bar.complete",
    finished_style: StyleType = "bar.finished",
    pulse_style: StyleType = "bar.pulse",
    table_column: Optional[Column] = None,
) -> None:
    self.bar_width = bar_width
    self.style = style
    self.complete_style = complete_style
    self.finished_style = finished_style
    self.pulse_style = pulse_style
    super().__init__(table_column=table_column)

def render(self, task: "Task") -> ProgressBar:
    """Gets a progress bar widget for a task."""
    return ProgressBar(
        total=max(0, task.total) if task.total is not None else None,
        completed=max(0, task.completed),
        width=None if self.bar_width is None else max(1, self.bar_width),
        pulse=not task.started,
        animation_time=task.get_time(),
        style=self.style,
        complete_style=self.complete_style,
        finished_style=self.finished_style,
        pulse_style=self.pulse_style,
    )


