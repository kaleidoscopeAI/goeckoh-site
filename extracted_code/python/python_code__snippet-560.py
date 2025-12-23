def _legacy_key(s):
    def get_parts(s):
        result = []
        for p in _VERSION_PART.split(s.lower()):
            p = _VERSION_REPLACE.get(p, p)
            if p:
                if '0' <= p[:1] <= '9':
                    p = p.zfill(8)
                else:
                    p = '*' + p
                result.append(p)
        result.append('*final')
        return result

    result = []
    for p in get_parts(s):
        if p.startswith('*'):
            if p < '*final':
                while result and result[-1] == '*final-':
                    result.pop()
            while result and result[-1] == '00000000':
                result.pop()
        result.append(p)
    return tuple(result)


class LegacyVersion(Version):
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


class LegacyMatcher(Matcher):
    version_class = LegacyVersion

    _operators = dict(Matcher._operators)
    _operators['~='] = '_match_compatible'

    numeric_re = re.compile(r'^(\d+(\.\d+)*)')

    def _match_compatible(self, version, constraint, prefix):
        if version < constraint:
            return False
        m = self.numeric_re.match(str(constraint))
        if not m:
            logger.warning('Cannot compute compatible match for version %s '
                           ' and constraint %s', version, constraint)
            return True
        s = m.groups()[0]
        if '.' in s:
            s = s.rsplit('.', 1)[0]
        return _match_prefix(version, s)

