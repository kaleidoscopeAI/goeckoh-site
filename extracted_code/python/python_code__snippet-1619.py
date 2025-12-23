"""A column containing text."""

def __init__(
    self,
    text_format: str,
    style: StyleType = "none",
    justify: JustifyMethod = "left",
    markup: bool = True,
    highlighter: Optional[Highlighter] = None,
    table_column: Optional[Column] = None,
) -> None:
    self.text_format = text_format
    self.justify: JustifyMethod = justify
    self.style = style
    self.markup = markup
    self.highlighter = highlighter
    super().__init__(table_column=table_column or Column(no_wrap=True))

def render(self, task: "Task") -> Text:
    _text = self.text_format.format(task=task)
    if self.markup:
        text = Text.from_markup(_text, style=self.style, justify=self.justify)
    else:
        text = Text(_text, style=self.style, justify=self.justify)
    if self.highlighter:
        self.highlighter.highlight(text)
    return text


