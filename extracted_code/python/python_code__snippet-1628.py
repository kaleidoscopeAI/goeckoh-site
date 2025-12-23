"""Renders human readable transfer speed."""

def render(self, task: "Task") -> Text:
    """Show data transfer speed."""
    speed = task.finished_speed or task.speed
    if speed is None:
        return Text("?", style="progress.data.speed")
    data_speed = filesize.decimal(int(speed))
    return Text(f"{data_speed}/s", style="progress.data.speed")


