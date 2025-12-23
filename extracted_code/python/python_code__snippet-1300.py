"""Raised if a hook is missing and we are not executing the fallback"""
def __init__(self, hook_name=None):
    super().__init__(hook_name)
    self.hook_name = hook_name


