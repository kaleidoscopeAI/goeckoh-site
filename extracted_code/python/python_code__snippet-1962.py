default_time_format = "%Y-%m-%dT%H:%M:%S"

def __init__(
    self,
    *args: Any,
    add_timestamp: bool = False,
    **kwargs: Any,
) -> None:
    """
    A logging.Formatter that obeys the indent_log() context manager.

    :param add_timestamp: A bool indicating output lines should be prefixed
        with their record's timestamp.
    """
    self.add_timestamp = add_timestamp
    super().__init__(*args, **kwargs)

def get_message_start(self, formatted: str, levelno: int) -> str:
    """
    Return the start of the formatted log message (not counting the
    prefix to add to each line).
    """
    if levelno < logging.WARNING:
        return ""
    if formatted.startswith(DEPRECATION_MSG_PREFIX):
        # Then the message already has a prefix.  We don't want it to
        # look like "WARNING: DEPRECATION: ...."
        return ""
    if levelno < logging.ERROR:
        return "WARNING: "

    return "ERROR: "

def format(self, record: logging.LogRecord) -> str:
    """
    Calls the standard formatter, but will indent all of the log message
    lines by our current indentation level.
    """
    formatted = super().format(record)
    message_start = self.get_message_start(formatted, record.levelno)
    formatted = message_start + formatted

    prefix = ""
    if self.add_timestamp:
        prefix = f"{self.formatTime(record)} "
    prefix += " " * get_indentation()
    formatted = "".join([prefix + line for line in formatted.splitlines(True)])
    return formatted


