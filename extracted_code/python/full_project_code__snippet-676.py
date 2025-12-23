"""Matches if the current position is at the end of a :class:`Word`,
and is not followed by any character in a given set of ``word_chars``
(default= ``printables``). To emulate the ``\b`` behavior of
regular expressions, use ``WordEnd(alphanums)``. ``WordEnd``
will also match at the end of the string being parsed, or at the end
of a line.
"""

def __init__(self, word_chars: str = printables, *, wordChars: str = printables):
    wordChars = word_chars if wordChars == printables else wordChars
    super().__init__()
    self.wordChars = set(wordChars)
    self.skipWhitespace = False
    self.errmsg = "Not at the end of a word"

def parseImpl(self, instring, loc, doActions=True):
    instrlen = len(instring)
    if instrlen > 0 and loc < instrlen:
        if (
            instring[loc] in self.wordChars
            or instring[loc - 1] not in self.wordChars
        ):
            raise ParseException(instring, loc, self.errmsg, self)
    return loc, []


