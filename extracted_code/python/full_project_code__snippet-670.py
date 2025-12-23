"""A short-cut class for defining :class:`Word` ``(characters, exact=1)``,
when defining a match of any single character in a string of
characters.
"""

def __init__(
    self,
    charset: str,
    as_keyword: bool = False,
    exclude_chars: typing.Optional[str] = None,
    *,
    asKeyword: bool = False,
    excludeChars: typing.Optional[str] = None,
):
    asKeyword = asKeyword or as_keyword
    excludeChars = excludeChars or exclude_chars
    super().__init__(
        charset, exact=1, as_keyword=asKeyword, exclude_chars=excludeChars
    )


