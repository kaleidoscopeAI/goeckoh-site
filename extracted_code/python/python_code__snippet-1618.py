"""A column with a 'spinner' animation.

Args:
    spinner_name (str, optional): Name of spinner animation. Defaults to "dots".
    style (StyleType, optional): Style of spinner. Defaults to "progress.spinner".
    speed (float, optional): Speed factor of spinner. Defaults to 1.0.
    finished_text (TextType, optional): Text used when task is finished. Defaults to " ".
"""

def __init__(
    self,
    spinner_name: str = "dots",
    style: Optional[StyleType] = "progress.spinner",
    speed: float = 1.0,
    finished_text: TextType = " ",
    table_column: Optional[Column] = None,
):
    self.spinner = Spinner(spinner_name, style=style, speed=speed)
    self.finished_text = (
        Text.from_markup(finished_text)
        if isinstance(finished_text, str)
        else finished_text
    )
    super().__init__(table_column=table_column)

def set_spinner(
    self,
    spinner_name: str,
    spinner_style: Optional[StyleType] = "progress.spinner",
    speed: float = 1.0,
) -> None:
    """Set a new spinner.

    Args:
        spinner_name (str): Spinner name, see python -m rich.spinner.
        spinner_style (Optional[StyleType], optional): Spinner style. Defaults to "progress.spinner".
        speed (float, optional): Speed factor of spinner. Defaults to 1.0.
    """
    self.spinner = Spinner(spinner_name, style=spinner_style, speed=speed)

def render(self, task: "Task") -> RenderableType:
    text = (
        self.finished_text
        if task.finished
        else self.spinner.render(task.get_time())
    )
    return text


