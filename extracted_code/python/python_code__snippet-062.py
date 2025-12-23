def is_semver(s):
    return _SEMVER_RE.match(s)


def _semantic_key(s):
    def make_tuple(s, absent):
        if s is None:
            result = (absent,)
        else:
            parts = s[1:].split('.')
            # We can't compare ints and strings on Python 3, so fudge it
            # by zero-filling numeric values so simulate a numeric comparison
            result = tuple([p.zfill(8) if p.isdigit() else p for p in parts])
        return result

    m = is_semver(s)
    if not m:
        raise UnsupportedVersionError(s)
    groups = m.groups()
    major, minor, patch = [int(i) for i in groups[:3]]
    # choose the '|' and '*' so that versions sort correctly
    pre, build = make_tuple(groups[3], '|'), make_tuple(groups[5], '*')
    return (major, minor, patch), pre, build


class SemanticVersion(Version):
    def parse(self, s):
        return _semantic_key(s)

    @property
    def is_prerelease(self):
        return self._parts[1][0] != '|'


class SemanticMatcher(Matcher):
    version_class = SemanticVersion


class VersionScheme(object):
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


