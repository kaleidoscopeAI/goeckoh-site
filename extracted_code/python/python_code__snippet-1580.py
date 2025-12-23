r"""Print object(s) supplied via positional arguments.
This function has an identical signature to the built-in print.
For more advanced features, see the :class:`~rich.console.Console` class.

Args:
    sep (str, optional): Separator between printed objects. Defaults to " ".
    end (str, optional): Character to write at end of output. Defaults to "\\n".
    file (IO[str], optional): File to write to, or None for stdout. Defaults to None.
    flush (bool, optional): Has no effect as Rich always flushes output. Defaults to False.

"""
from .console import Console

write_console = get_console() if file is None else Console(file=file)
return write_console.print(*objects, sep=sep, end=end)


