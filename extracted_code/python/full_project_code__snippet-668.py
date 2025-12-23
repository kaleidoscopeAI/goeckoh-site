"""
Caseless version of :class:`Keyword`.

Example::

    CaselessKeyword("CMD")[1, ...].parse_string("cmd CMD Cmd10")
    # -> ['CMD', 'CMD']

(Contrast with example for :class:`CaselessLiteral`.)
"""

def __init__(
    self,
    match_string: str = "",
    ident_chars: typing.Optional[str] = None,
    *,
    matchString: str = "",
    identChars: typing.Optional[str] = None,
):
    identChars = identChars or ident_chars
    match_string = matchString or match_string
    super().__init__(match_string, identChars, caseless=True)


