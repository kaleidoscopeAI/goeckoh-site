"""Multiple HashError instances rolled into one for reporting"""

def __init__(self) -> None:
    self.errors: List["HashError"] = []

def append(self, error: "HashError") -> None:
    self.errors.append(error)

def __str__(self) -> str:
    lines = []
    self.errors.sort(key=lambda e: e.order)
    for cls, errors_of_cls in groupby(self.errors, lambda e: e.__class__):
        lines.append(cls.head)
        lines.extend(e.body() for e in errors_of_cls)
    if lines:
        return "\n".join(lines)
    return ""

def __bool__(self) -> bool:
    return bool(self.errors)


