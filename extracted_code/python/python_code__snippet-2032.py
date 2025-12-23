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


