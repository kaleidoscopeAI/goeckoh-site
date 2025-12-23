def __init__(self, criterion):
    super(RequirementsConflicted, self).__init__(criterion)
    self.criterion = criterion

def __str__(self):
    return "Requirements conflict: {}".format(
        ", ".join(repr(r) for r in self.criterion.iter_requirement()),
    )


