def __init__(self, requirement_string):
    """DO NOT CALL THIS UNDOCUMENTED METHOD; use Requirement.parse()!"""
    super(Requirement, self).__init__(requirement_string)
    self.unsafe_name = self.name
    project_name = safe_name(self.name)
    self.project_name, self.key = project_name, project_name.lower()
    self.specs = [(spec.operator, spec.version) for spec in self.specifier]
    self.extras = tuple(map(safe_extra, self.extras))
    self.hashCmp = (
        self.key,
        self.url,
        self.specifier,
        frozenset(self.extras),
        str(self.marker) if self.marker else None,
    )
    self.__hash = hash(self.hashCmp)

def __eq__(self, other):
    return isinstance(other, Requirement) and self.hashCmp == other.hashCmp

def __ne__(self, other):
    return not self == other

def __contains__(self, item):
    if isinstance(item, Distribution):
        if item.key != self.key:
            return False

        item = item.version

    # Allow prereleases always in order to match the previous behavior of
    # this method. In the future this should be smarter and follow PEP 440
    # more accurately.
    return self.specifier.contains(item, prereleases=True)

def __hash__(self):
    return self.__hash

def __repr__(self):
    return "Requirement.parse(%r)" % str(self)

@staticmethod
def parse(s):
    (req,) = parse_requirements(s)
    return req


