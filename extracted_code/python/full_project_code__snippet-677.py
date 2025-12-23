"""
Expression to match one or more expressions at a given indentation level.
Useful for parsing text where structure is implied by indentation (like Python source code).
"""

class _Indent(Empty):
    def __init__(self, ref_col: int):
        super().__init__()
        self.errmsg = f"expected indent at column {ref_col}"
        self.add_condition(lambda s, l, t: col(l, s) == ref_col)

class _IndentGreater(Empty):
    def __init__(self, ref_col: int):
        super().__init__()
        self.errmsg = f"expected indent at column greater than {ref_col}"
        self.add_condition(lambda s, l, t: col(l, s) > ref_col)

def __init__(
    self, expr: ParserElement, *, recursive: bool = False, grouped: bool = True
):
    super().__init__(expr, savelist=True)
    # if recursive:
    #     raise NotImplementedError("IndentedBlock with recursive is not implemented")
    self._recursive = recursive
    self._grouped = grouped
    self.parent_anchor = 1

def parseImpl(self, instring, loc, doActions=True):
    # advance parse position to non-whitespace by using an Empty()
    # this should be the column to be used for all subsequent indented lines
    anchor_loc = Empty().preParse(instring, loc)

    # see if self.expr matches at the current location - if not it will raise an exception
    # and no further work is necessary
    self.expr.try_parse(instring, anchor_loc, do_actions=doActions)

    indent_col = col(anchor_loc, instring)
    peer_detect_expr = self._Indent(indent_col)

    inner_expr = Empty() + peer_detect_expr + self.expr
    if self._recursive:
        sub_indent = self._IndentGreater(indent_col)
        nested_block = IndentedBlock(
            self.expr, recursive=self._recursive, grouped=self._grouped
        )
        nested_block.set_debug(self.debug)
        nested_block.parent_anchor = indent_col
        inner_expr += Opt(sub_indent + nested_block)

    inner_expr.set_name(f"inner {hex(id(inner_expr))[-4:].upper()}@{indent_col}")
    block = OneOrMore(inner_expr)

    trailing_undent = self._Indent(self.parent_anchor) | StringEnd()

    if self._grouped:
        wrapper = Group
    else:
        wrapper = lambda expr: expr
    return (wrapper(block) + Optional(trailing_undent)).parseImpl(
        instring, anchor_loc, doActions
    )


