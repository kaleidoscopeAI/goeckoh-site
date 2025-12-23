def parseImpl(self, instring, loc, doActions=True):
    if instring[loc] == self.firstMatchChar:
        return loc + 1, self.match
    raise ParseException(instring, loc, self.errmsg, self)


