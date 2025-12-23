def __init__(self, key, matcher, suggester=None):
    self.key = key
    self.matcher = matcher
    self.suggester = suggester

def is_valid_version(self, s):
    try:
        self.matcher.version_class(s)
        result = True
    except UnsupportedVersionError:
        result = False
    return result

def is_valid_matcher(self, s):
    try:
        self.matcher(s)
        result = True
    except UnsupportedVersionError:
        result = False
    return result

def is_valid_constraint_list(self, s):
    """
    Used for processing some metadata fields
    """
    # See issue #140. Be tolerant of a single trailing comma.
    if s.endswith(','):
        s = s[:-1]
    return self.is_valid_matcher('dummy_name (%s)' % s)

def suggest(self, s):
    if self.suggester is None:
        result = None
    else:
        result = self.suggester(s)
    return result


