"""May be raised by build_sdist if the backend indicates that it can't."""
def __init__(self, traceback):
    self.traceback = traceback


