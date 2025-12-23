"""A requested distribution was not found"""

_template = (
    "The '{self.req}' distribution was not found "
    "and is required by {self.requirers_str}"
)

@property
def req(self):
    return self.args[0]

@property
def requirers(self):
    return self.args[1]

@property
def requirers_str(self):
    if not self.requirers:
        return 'the application'
    return ', '.join(self.requirers)

def report(self):
    return self._template.format(**locals())

def __str__(self):
    return self.report()


