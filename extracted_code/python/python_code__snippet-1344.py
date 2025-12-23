"""A rational version.

Good:
    1.2         # equivalent to "1.2.0"
    1.2.0
    1.2a1
    1.2.3a2
    1.2.3b1
    1.2.3c1
    1.2.3.4
    TODO: fill this out

Bad:
    1           # minimum two numbers
    1.2a        # release level must have a release serial
    1.2.3b
"""
def parse(self, s):
    result = _normalized_key(s)
    # _normalized_key loses trailing zeroes in the release
    # clause, since that's needed to ensure that X.Y == X.Y.0 == X.Y.0.0
    # However, PEP 440 prefix matching needs it: for example,
    # (~= 1.4.5.0) matches differently to (~= 1.4.5.0.0).
    m = PEP440_VERSION_RE.match(s)      # must succeed
    groups = m.groups()
    self._release_clause = tuple(int(v) for v in groups[1].split('.'))
    return result

PREREL_TAGS = set(['a', 'b', 'c', 'rc', 'dev'])

@property
def is_prerelease(self):
    return any(t[0] in self.PREREL_TAGS for t in self._parts if t)


