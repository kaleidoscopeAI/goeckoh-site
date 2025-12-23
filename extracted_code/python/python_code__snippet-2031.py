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


