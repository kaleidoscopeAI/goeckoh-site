"""Renders completed filesize."""

def render(self, task: "Task") -> Text:
    """Show data completed."""
    data_size = filesize.decimal(int(task.completed))
    return Text(data_size, style="progress.filesize")


