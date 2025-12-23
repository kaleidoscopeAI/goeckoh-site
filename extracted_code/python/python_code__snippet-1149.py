def issue_warning(*args, **kw):
    level = 1
    g = globals()
    try:
        # find the first stack frame that is *not* code in
        # the pkg_resources module, to use for the warning
        while sys._getframe(level).f_globals is g:
            level += 1
    except ValueError:
        pass
    warnings.warn(stacklevel=level + 1, *args, **kw)


def parse_requirements(strs):
    """
    Yield ``Requirement`` objects for each specification in `strs`.

    `strs` must be a string, or a (possibly-nested) iterable thereof.
    """
    return map(Requirement, join_continuation(map(drop_comment, yield_lines(strs))))


class RequirementParseError(packaging.requirements.InvalidRequirement):
    "Compatibility wrapper for InvalidRequirement"


class Requirement(packaging.requirements.Requirement):
    def __init__(self, requirement_string):
        """DO NOT CALL THIS UNDOCUMENTED METHOD; use Requirement.parse()!"""
        super(Requirement, self).__init__(requirement_string)
        self.unsafe_name = self.name
        project_name = safe_name(self.name)
        self.project_name, self.key = project_name, project_name.lower()
        self.specs = [(spec.operator, spec.version) for spec in self.specifier]
        self.extras = tuple(map(safe_extra, self.extras))
        self.hashCmp = (
            self.key,
            self.url,
            self.specifier,
            frozenset(self.extras),
            str(self.marker) if self.marker else None,
        )
        self.__hash = hash(self.hashCmp)

    def __eq__(self, other):
        return isinstance(other, Requirement) and self.hashCmp == other.hashCmp

    def __ne__(self, other):
        return not self == other

    def __contains__(self, item):
        if isinstance(item, Distribution):
            if item.key != self.key:
                return False

            item = item.version

        # Allow prereleases always in order to match the previous behavior of
        # this method. In the future this should be smarter and follow PEP 440
        # more accurately.
        return self.specifier.contains(item, prereleases=True)

    def __hash__(self):
        return self.__hash

    def __repr__(self):
        return "Requirement.parse(%r)" % str(self)

    @staticmethod
    def parse(s):
        (req,) = parse_requirements(s)
        return req


def _always_object(classes):
    """
    Ensure object appears in the mro even
    for old-style classes.
    """
    if object not in classes:
        return classes + (object,)
    return classes


def _find_adapter(registry, ob):
    """Return an adapter factory for `ob` from `registry`"""
    types = _always_object(inspect.getmro(getattr(ob, '__class__', type(ob))))
    for t in types:
        if t in registry:
            return registry[t]


def ensure_directory(path):
    """Ensure that the parent directory of `path` exists"""
    dirname = os.path.dirname(path)
    os.makedirs(dirname, exist_ok=True)


def _bypass_ensure_directory(path):
    """Sandbox-bypassing version of ensure_directory()"""
    if not WRITE_SUPPORT:
        raise IOError('"os.mkdir" not supported on this platform.')
    dirname, filename = split(path)
    if dirname and filename and not isdir(dirname):
        _bypass_ensure_directory(dirname)
        try:
            mkdir(dirname, 0o755)
        except FileExistsError:
            pass


def split_sections(s):
    """Split a string or iterable thereof into (section, content) pairs

    Each ``section`` is a stripped version of the section header ("[section]")
    and each ``content`` is a list of stripped lines excluding blank lines and
    comment-only lines.  If there are any such lines before the first section
    header, they're returned in a first ``section`` of ``None``.
    """
    section = None
    content = []
    for line in yield_lines(s):
        if line.startswith("["):
            if line.endswith("]"):
                if section or content:
                    yield section, content
                section = line[1:-1].strip()
                content = []
            else:
                raise ValueError("Invalid section heading", line)
        else:
            content.append(line)

    # wrap up last segment
    yield section, content


def _mkstemp(*args, **kw):
    old_open = os.open
    try:
        # temporarily bypass sandboxing
        os.open = os_open
        return tempfile.mkstemp(*args, **kw)
    finally:
        # and then put it back
        os.open = old_open


