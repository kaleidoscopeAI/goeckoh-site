"""
An already-installed version conflicts with the requested version.

Should be initialized with the installed Distribution and the requested
Requirement.
"""

_template = "{self.dist} is installed but {self.req} is required"

@property
def dist(self):
    return self.args[0]

@property
def req(self):
    return self.args[1]

def report(self):
    return self._template.format(**locals())

def with_context(self, required_by):
    """
    If required_by is non-empty, return a version of self that is a
    ContextualVersionConflict.
    """
    if not required_by:
        return self
    args = self.args + (required_by,)
    return ContextualVersionConflict(*args)


