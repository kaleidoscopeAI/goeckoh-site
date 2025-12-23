class NormalizedVersion(Version):
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


def _match_prefix(x, y):
    x = str(x)
    y = str(y)
    if x == y:
        return True
    if not x.startswith(y):
        return False
    n = len(y)
    return x[n] == '.'


class NormalizedMatcher(Matcher):
    version_class = NormalizedVersion

    # value is either a callable or the name of a method
    _operators = {
        '~=': '_match_compatible',
        '<': '_match_lt',
        '>': '_match_gt',
        '<=': '_match_le',
        '>=': '_match_ge',
        '==': '_match_eq',
        '===': '_match_arbitrary',
        '!=': '_match_ne',
    }

    def _adjust_local(self, version, constraint, prefix):
        if prefix:
            strip_local = '+' not in constraint and version._parts[-1]
        else:
            # both constraint and version are
            # NormalizedVersion instances.
            # If constraint does not have a local component,
            # ensure the version doesn't, either.
            strip_local = not constraint._parts[-1] and version._parts[-1]
        if strip_local:
            s = version._string.split('+', 1)[0]
            version = self.version_class(s)
        return version, constraint

    def _match_lt(self, version, constraint, prefix):
        version, constraint = self._adjust_local(version, constraint, prefix)
        if version >= constraint:
            return False
        release_clause = constraint._release_clause
        pfx = '.'.join([str(i) for i in release_clause])
        return not _match_prefix(version, pfx)

    def _match_gt(self, version, constraint, prefix):
        version, constraint = self._adjust_local(version, constraint, prefix)
        if version <= constraint:
            return False
        release_clause = constraint._release_clause
        pfx = '.'.join([str(i) for i in release_clause])
        return not _match_prefix(version, pfx)

    def _match_le(self, version, constraint, prefix):
        version, constraint = self._adjust_local(version, constraint, prefix)
        return version <= constraint

    def _match_ge(self, version, constraint, prefix):
        version, constraint = self._adjust_local(version, constraint, prefix)
        return version >= constraint

    def _match_eq(self, version, constraint, prefix):
        version, constraint = self._adjust_local(version, constraint, prefix)
        if not prefix:
            result = (version == constraint)
        else:
            result = _match_prefix(version, constraint)
        return result

    def _match_arbitrary(self, version, constraint, prefix):
        return str(version) == str(constraint)

    def _match_ne(self, version, constraint, prefix):
        version, constraint = self._adjust_local(version, constraint, prefix)
        if not prefix:
            result = (version != constraint)
        else:
            result = not _match_prefix(version, constraint)
        return result

    def _match_compatible(self, version, constraint, prefix):
        version, constraint = self._adjust_local(version, constraint, prefix)
        if version == constraint:
            return True
        if version < constraint:
            return False
