def __init__(self, name, prefix, suffix, flags):
    self.name = name
    self.prefix = prefix
    self.suffix = suffix
    self.flags = flags

@cached_property
def value(self):
    return resolve(self.prefix, self.suffix)

def __repr__(self):  # pragma: no cover
    return '<ExportEntry %s = %s:%s %s>' % (self.name, self.prefix,
                                            self.suffix, self.flags)

def __eq__(self, other):
    if not isinstance(other, ExportEntry):
        result = False
    else:
        result = (self.name == other.name and self.prefix == other.prefix
                  and self.suffix == other.suffix
                  and self.flags == other.flags)
    return result

__hash__ = object.__hash__


