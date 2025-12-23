class ParsedRequirement:
    def __init__(
        self,
        requirement: str,
        is_editable: bool,
        comes_from: str,
        constraint: bool,
        options: Optional[Dict[str, Any]] = None,
        line_source: Optional[str] = None,
    ) -> None:
        self.requirement = requirement
        self.is_editable = is_editable
        self.comes_from = comes_from
        self.options = options
        self.constraint = constraint
        self.line_source = line_source


class ParsedLine:
    def __init__(
        self,
        filename: str,
        lineno: int,
        args: str,
        opts: Values,
        constraint: bool,
    ) -> None:
        self.filename = filename
        self.lineno = lineno
        self.opts = opts
        self.constraint = constraint

        if args:
            self.is_requirement = True
            self.is_editable = False
            self.requirement = args
        elif opts.editables:
            self.is_requirement = True
            self.is_editable = True
            # We don't support multiple -e on one line
            self.requirement = opts.editables[0]
        else:
            self.is_requirement = False


def parse_requirements(
    filename: str,
    session: PipSession,
    finder: Optional["PackageFinder"] = None,
    options: Optional[optparse.Values] = None,
    constraint: bool = False,
