"""
Optional repetition of zero or more of the given expression.

Parameters:

- ``expr`` - expression that must match zero or more times
- ``stop_on`` - expression for a terminating sentinel
  (only required if the sentinel would ordinarily match the repetition
  expression) - (default= ``None``)

Example: similar to :class:`OneOrMore`
"""

def __init__(
    self,
    expr: Union[str, ParserElement],
    stop_on: typing.Optional[Union[ParserElement, str]] = None,
    *,
    stopOn: typing.Optional[Union[ParserElement, str]] = None,
):
    super().__init__(expr, stopOn=stopOn or stop_on)
    self.mayReturnEmpty = True

def parseImpl(self, instring, loc, doActions=True):
    try:
        return super().parseImpl(instring, loc, doActions)
    except (ParseException, IndexError):
        return loc, ParseResults([], name=self.resultsName)

def _generateDefaultName(self) -> str:
    return "[" + str(self.expr) + "]..."


