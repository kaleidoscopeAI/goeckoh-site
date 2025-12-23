"""Matches if the current position is at the beginning of a
:class:`Word`, and is not preceded by any character in a given
set of ``word_chars`` (default= ``printables``). To emulate the
``\b`` behavior of regular expressions, use
``WordStart(alphanums)``. ``WordStart`` will also match at
the beginning of the string being parsed, or at the beginning of
a line.
"""

def __init__(self, word_chars: str = printables, *, wordChars: str = printables):
    wordChars = word_chars if wordChars == printables else wordChars
    super().__init__()
    self.wordChars = set(wordChars)
    self.errmsg = "Not at the start of a word"

def parseImpl(self, instring, loc, doActions=True):
    if loc != 0:
        if (
            instring[loc - 1] in self.wordChars
            or instring[loc] not in self.wordChars
        ):
            raise ParseException(instring, loc, self.errmsg, self)
    return loc, []


