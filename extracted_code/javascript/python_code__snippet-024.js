"""A convenience function for pretty printing.

Args:
    _object (Any): Object to pretty print.
    console (Console, optional): Console instance, or None to use default. Defaults to None.
    max_length (int, optional): Maximum length of containers before abbreviating, or None for no abbreviation.
        Defaults to None.
    max_string (int, optional): Maximum length of strings before truncating, or None to disable. Defaults to None.
    max_depth (int, optional): Maximum depth for nested data structures, or None for unlimited depth. Defaults to None.
    indent_guides (bool, optional): Enable indentation guides. Defaults to True.
    expand_all (bool, optional): Expand all containers. Defaults to False.
"""
_console = get_console() if console is None else console
_console.print(
    Pretty(
        _object,
        max_length=max_length,
        max_string=max_string,
        max_depth=max_depth,
        indent_guides=indent_guides,
        expand_all=expand_all,
        overflow="ignore",
    ),
    soft_wrap=True,
)


