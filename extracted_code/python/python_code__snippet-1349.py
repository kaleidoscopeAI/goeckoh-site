def parse(self, s):
    return _legacy_key(s)

@property
def is_prerelease(self):
    result = False
    for x in self._parts:
        if (isinstance(x, string_types) and x.startswith('*') and
                x < '*final'):
            result = True
            break
    return result


