def parse(self, s):
    return _semantic_key(s)

@property
def is_prerelease(self):
    return self._parts[1][0] != '|'


