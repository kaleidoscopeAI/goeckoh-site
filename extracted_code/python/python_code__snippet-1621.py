"""Renders time elapsed."""

def render(self, task: "Task") -> Text:
    """Show time elapsed."""
    elapsed = task.finished_time if task.finished else task.elapsed
    if elapsed is None:
        return Text("-:--:--", style="progress.elapsed")
    delta = timedelta(seconds=int(elapsed))
    return Text(str(delta), style="progress.elapsed")


