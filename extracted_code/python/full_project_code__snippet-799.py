REPR_FIELDS = ("sleep",)
NAME = "retry"

def __init__(self, sleep: t.SupportsFloat) -> None:
    self.sleep = float(sleep)


