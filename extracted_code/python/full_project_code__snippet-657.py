tup: Tuple["ParseResults", int]
__slots__ = ["tup"]

def __init__(self, p1: "ParseResults", p2: int):
    self.tup: Tuple[ParseResults, int] = (p1, p2)

def __getitem__(self, i):
    return self.tup[i]

def __getstate__(self):
    return self.tup

def __setstate__(self, *args):
    self.tup = args[0]


