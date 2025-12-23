"""Matches if expression matches at the beginning of the parse
string::

    AtStringStart(Word(nums)).parse_string("123")
    # prints ["123"]

    AtStringStart(Word(nums)).parse_string("    123")
    # raises ParseException
"""

def __init__(self, expr: Union[ParserElement, str]):
    super().__init__(expr)
    self.callPreparse = False

def parseImpl(self, instring, loc, doActions=True):
    if loc != 0:
        raise ParseException(instring, loc, "not found at string start")
    return super().parseImpl(instring, loc, doActions)


