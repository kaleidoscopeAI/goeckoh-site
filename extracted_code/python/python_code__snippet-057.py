class UnsupportedVersionError(ValueError):
    """This is an unsupported version."""
    pass


class Version(object):
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


class Matcher(object):
    version_class = None

    # value is either a callable or the name of a method
    _operators = {
        '<': lambda v, c, p: v < c,
        '>': lambda v, c, p: v > c,
        '<=': lambda v, c, p: v == c or v < c,
        '>=': lambda v, c, p: v == c or v > c,
        '==': lambda v, c, p: v == c,
        '===': lambda v, c, p: v == c,
        # by default, compatible => >=.
        '~=': lambda v, c, p: v == c or v > c,
        '!=': lambda v, c, p: v != c,
    }

    # this is a method only to support alternative implementations
    # via overriding
    def parse_requirement(self, s):
        return parse_requirement(s)

    def __init__(self, s):
        if self.version_class is None:
            raise ValueError('Please specify a version class')
        self._string = s = s.strip()
        r = self.parse_requirement(s)
        if not r:
            raise ValueError('Not valid: %r' % s)
        self.name = r.name
        self.key = self.name.lower()    # for case-insensitive comparisons
        clist = []
        if r.constraints:
            # import pdb; pdb.set_trace()
            for op, s in r.constraints:
                if s.endswith('.*'):
                    if op not in ('==', '!='):
                        raise ValueError('\'.*\' not allowed for '
                                         '%r constraints' % op)
                    # Could be a partial version (e.g. for '2.*') which
                    # won't parse as a version, so keep it as a string
                    vn, prefix = s[:-2], True
                    # Just to check that vn is a valid version
                    self.version_class(vn)
                else:
                    # Should parse as a version, so we can create an
                    # instance for the comparison
                    vn, prefix = self.version_class(s), False
                clist.append((op, vn, prefix))
        self._parts = tuple(clist)

    def match(self, version):
        """
        Check if the provided version matches the constraints.

        :param version: The version to match against this instance.
        :type version: String or :class:`Version` instance.
        """
        if isinstance(version, string_types):
            version = self.version_class(version)
        for operator, constraint, prefix in self._parts:
            f = self._operators.get(operator)
            if isinstance(f, string_types):
                f = getattr(self, f)
            if not f:
                msg = ('%r not implemented '
                       'for %s' % (operator, self.__class__.__name__))
                raise NotImplementedError(msg)
            if not f(version, constraint, prefix):
                return False
        return True

    @property
    def exact_version(self):
        result = None
        if len(self._parts) == 1 and self._parts[0][0] in ('==', '==='):
            result = self._parts[0][1]
        return result

    def _check_compatible(self, other):
        if type(self) != type(other) or self.name != other.name:
            raise TypeError('cannot compare %s and %s' % (self, other))

    def __eq__(self, other):
        self._check_compatible(other)
        return self.key == other.key and self._parts == other._parts

    def __ne__(self, other):
        return not self.__eq__(other)

    # See http://docs.python.org/reference/datamodel#object.__hash__
    def __hash__(self):
        return hash(self.key) + hash(self._parts)

    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, self._string)

    def __str__(self):
        return self._string


