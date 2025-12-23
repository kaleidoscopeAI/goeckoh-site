"""
Abstract subclass of :class:`ParseExpression`, for converting parsed results.
"""

def __init__(self, expr: Union[ParserElement, str], savelist=False):
    super().__init__(expr)  # , savelist)
    self.saveAsList = False


