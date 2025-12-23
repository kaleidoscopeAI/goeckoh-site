"""Matches if current position is at the beginning of the parse
string
"""

def __init__(self):
    super().__init__()
    self.errmsg = "Expected start of text"

def parseImpl(self, instring, loc, doActions=True):
    if loc != 0:
        # see if entire string up to here is just whitespace and ignoreables
        if loc != self.preParse(instring, 0):
            raise ParseException(instring, loc, self.errmsg, self)
    return loc, []


