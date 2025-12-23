"""Abstract base class for highlighters."""

def __call__(self, text: Union[str, Text]) -> Text:
    """Highlight a str or Text instance.

    Args:
        text (Union[str, ~Text]): Text to highlight.

    Raises:
        TypeError: If not called with text or str.

    Returns:
        Text: A test instance with highlighting applied.
    """
    if isinstance(text, str):
        highlight_text = Text(text)
    elif isinstance(text, Text):
        highlight_text = text.copy()
    else:
        raise TypeError(f"str or Text instance required, not {text!r}")
    self.highlight(highlight_text)
    return highlight_text

@abstractmethod
def highlight(self, text: Text) -> None:
    """Apply highlighting in place to text.

    Args:
        text (~Text): A text object highlight.
    """


