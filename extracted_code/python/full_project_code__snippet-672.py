"""Matches if current position is at the end of a line within the
parse string
"""

def __init__(self):
    super().__init__()
    self.whiteChars.discard("\n")
    self.set_whitespace_chars(self.whiteChars, copy_defaults=False)
    self.errmsg = "Expected end of line"

def parseImpl(self, instring, loc, doActions=True):
    if loc < len(instring):
        if instring[loc] == "\n":
            return loc + 1, "\n"
        else:
            raise ParseException(instring, loc, self.errmsg, self)
    elif loc == len(instring):
        return loc + 1, []
    else:
        raise ParseException(instring, loc, self.errmsg, self)


