"""Renders estimated time remaining.

Args:
    compact (bool, optional): Render MM:SS when time remaining is less than an hour. Defaults to False.
    elapsed_when_finished (bool, optional): Render time elapsed when the task is finished. Defaults to False.
"""

# Only refresh twice a second to prevent jitter
max_refresh = 0.5

def __init__(
    self,
    compact: bool = False,
    elapsed_when_finished: bool = False,
    table_column: Optional[Column] = None,
):
    self.compact = compact
    self.elapsed_when_finished = elapsed_when_finished
    super().__init__(table_column=table_column)

def render(self, task: "Task") -> Text:
    """Show time remaining."""
    if self.elapsed_when_finished and task.finished:
        task_time = task.finished_time
        style = "progress.elapsed"
    else:
        task_time = task.time_remaining
        style = "progress.remaining"

    if task.total is None:
        return Text("", style=style)

    if task_time is None:
        return Text("--:--" if self.compact else "-:--:--", style=style)

    # Based on https://github.com/tqdm/tqdm/blob/master/tqdm/std.py
    minutes, seconds = divmod(int(task_time), 60)
    hours, minutes = divmod(minutes, 60)

    if self.compact and not hours:
        formatted = f"{minutes:02d}:{seconds:02d}"
    else:
        formatted = f"{hours:d}:{minutes:02d}:{seconds:02d}"

    return Text(formatted, style=style)


