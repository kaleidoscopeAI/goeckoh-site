def __init__(self, candidate, criterion):
    super(InconsistentCandidate, self).__init__(candidate, criterion)
    self.candidate = candidate
    self.criterion = criterion

def __str__(self):
    return "Provided candidate {!r} does not satisfy {}".format(
        self.candidate,
        ", ".join(repr(r) for r in self.criterion.iter_requirement()),
    )


