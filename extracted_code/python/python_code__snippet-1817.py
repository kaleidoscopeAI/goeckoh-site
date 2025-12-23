def __str__(self) -> str:
    before = ", ".join(str(a) for a in self.args[:-1])
    return f"Cannot set {before} and {self.args[-1]} together"


