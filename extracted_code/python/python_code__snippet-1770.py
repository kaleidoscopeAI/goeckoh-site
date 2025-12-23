"""
A VersionConflict that accepts a third parameter, the set of the
requirements that required the installed Distribution.
"""

_template = VersionConflict._template + ' by {self.required_by}'

@property
def required_by(self):
    return self.args[2]


