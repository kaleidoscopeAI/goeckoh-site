def __init__(self, s):
    self._string = s = s.strip()
    self._parts = parts = self.parse(s)
    assert isinstance(parts, tuple)
    assert len(parts) > 0

def parse(self, s):
    raise NotImplementedError('please implement in a subclass')

def _check_compatible(self, other):
    if type(self) != type(other):
        raise TypeError('cannot compare %r and %r' % (self, other))

def __eq__(self, other):
    self._check_compatible(other)
    return self._parts == other._parts

def __ne__(self, other):
    return not self.__eq__(other)

def __lt__(self, other):
    self._check_compatible(other)
    return self._parts < other._parts

def __gt__(self, other):
    return not (self.__lt__(other) or self.__eq__(other))

def __le__(self, other):
    return self.__lt__(other) or self.__eq__(other)

def __ge__(self, other):
    return self.__gt__(other) or self.__eq__(other)

# See http://docs.python.org/reference/datamodel#object.__hash__
def __hash__(self):
    return hash(self._parts)

def __repr__(self):
    return "%s('%s')" % (self.__class__.__name__, self._string)

def __str__(self):
    return self._string

@property
def is_prerelease(self):
    raise NotImplementedError('Please implement in subclasses.')


