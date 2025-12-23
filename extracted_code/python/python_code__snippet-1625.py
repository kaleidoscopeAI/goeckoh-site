"""Renders total filesize."""

def render(self, task: "Task") -> Text:
    """Show data completed."""
    data_size = filesize.decimal(int(task.total)) if task.total is not None else ""
    return Text(data_size, style="progress.filesize.total")


