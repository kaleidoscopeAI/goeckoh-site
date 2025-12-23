"""A variation on :class:`Literal` which matches "close" matches,
that is, strings with at most 'n' mismatching characters.
:class:`CloseMatch` takes parameters:

- ``match_string`` - string to be matched
- ``caseless`` - a boolean indicating whether to ignore casing when comparing characters
- ``max_mismatches`` - (``default=1``) maximum number of
  mismatches allowed to count as a match

The results from a successful parse will contain the matched text
from the input string and the following named results:

- ``mismatches`` - a list of the positions within the
  match_string where mismatches were found
- ``original`` - the original match_string used to compare
  against the input string

If ``mismatches`` is an empty list, then the match was an exact
match.

Example::

    patt = CloseMatch("ATCATCGAATGGA")
    patt.parse_string("ATCATCGAAXGGA") # -> (['ATCATCGAAXGGA'], {'mismatches': [[9]], 'original': ['ATCATCGAATGGA']})
    patt.parse_string("ATCAXCGAAXGGA") # -> Exception: Expected 'ATCATCGAATGGA' (with up to 1 mismatches) (at char 0), (line:1, col:1)

    # exact match
    patt.parse_string("ATCATCGAATGGA") # -> (['ATCATCGAATGGA'], {'mismatches': [[]], 'original': ['ATCATCGAATGGA']})

    # close match allowing up to 2 mismatches
    patt = CloseMatch("ATCATCGAATGGA", max_mismatches=2)
    patt.parse_string("ATCAXCGAAXGGA") # -> (['ATCAXCGAAXGGA'], {'mismatches': [[4, 9]], 'original': ['ATCATCGAATGGA']})
"""

def __init__(
    self,
    match_string: str,
    max_mismatches: typing.Optional[int] = None,
    *,
    maxMismatches: int = 1,
    caseless=False,
):
    maxMismatches = max_mismatches if max_mismatches is not None else maxMismatches
    super().__init__()
    self.match_string = match_string
    self.maxMismatches = maxMismatches
    self.errmsg = f"Expected {self.match_string!r} (with up to {self.maxMismatches} mismatches)"
    self.caseless = caseless
    self.mayIndexError = False
    self.mayReturnEmpty = False

def _generateDefaultName(self) -> str:
    return f"{type(self).__name__}:{self.match_string!r}"

def parseImpl(self, instring, loc, doActions=True):
    start = loc
    instrlen = len(instring)
    maxloc = start + len(self.match_string)

    if maxloc <= instrlen:
        match_string = self.match_string
        match_stringloc = 0
        mismatches = []
        maxMismatches = self.maxMismatches

        for match_stringloc, s_m in enumerate(
            zip(instring[loc:maxloc], match_string)
        ):
            src, mat = s_m
            if self.caseless:
                src, mat = src.lower(), mat.lower()

            if src != mat:
                mismatches.append(match_stringloc)
                if len(mismatches) > maxMismatches:
                    break
        else:
            loc = start + match_stringloc + 1
            results = ParseResults([instring[start:loc]])
            results["original"] = match_string
            results["mismatches"] = mismatches
            return loc, results

    raise ParseException(instring, loc, self.errmsg, self)


