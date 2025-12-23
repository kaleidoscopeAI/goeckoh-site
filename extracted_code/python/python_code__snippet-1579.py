"""Reconfigures the global console by replacing it with another.

Args:
    *args (Any): Positional arguments for the replacement :class:`~rich.console.Console`.
    **kwargs (Any): Keyword arguments for the replacement :class:`~rich.console.Console`.
"""
from pip._vendor.rich.console import Console

new_console = Console(*args, **kwargs)
_console = get_console()
_console.__dict__ = new_console.__dict__


