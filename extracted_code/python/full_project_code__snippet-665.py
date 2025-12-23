# internal placeholder class to hold a place were '...' is added to a parser element,
# once another ParserElement is added, this placeholder will be replaced with a SkipTo
def __init__(self, expr: ParserElement, must_skip: bool = False):
    super().__init__()
    self.anchor = expr
    self.must_skip = must_skip

def _generateDefaultName(self) -> str:
    return str(self.anchor + Empty()).replace("Empty", "...")

def __add__(self, other) -> "ParserElement":
    skipper = SkipTo(other).set_name("...")("_skipped*")
    if self.must_skip:

        def must_skip(t):
            if not t._skipped or t._skipped.as_list() == [""]:
                del t[0]
                t.pop("_skipped", None)

        def show_skip(t):
            if t._skipped.as_list()[-1:] == [""]:
                t.pop("_skipped")
                t["_skipped"] = "missing <" + repr(self.anchor) + ">"

        return (
            self.anchor + skipper().add_parse_action(must_skip)
            | skipper().add_parse_action(show_skip)
        ) + other

    return self.anchor + skipper + other

def __repr__(self):
    return self.defaultName

def parseImpl(self, *args):
    raise Exception(
        "use of `...` expression without following SkipTo target expression"
    )


