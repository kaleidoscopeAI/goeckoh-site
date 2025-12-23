"""Will be raised on missing hooks (if a fallback can't be used)."""
def __init__(self, hook_name):
    super().__init__(hook_name)
    self.hook_name = hook_name


