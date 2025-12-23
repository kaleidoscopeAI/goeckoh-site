"""Object representing an advertised importable object"""

def __init__(self, name, module_name, attrs=(), extras=(), dist=None):
    if not MODULE(module_name):
        raise ValueError("Invalid module name", module_name)
    self.name = name
    self.module_name = module_name
    self.attrs = tuple(attrs)
    self.extras = tuple(extras)
    self.dist = dist

def __str__(self):
    s = "%s = %s" % (self.name, self.module_name)
    if self.attrs:
        s += ':' + '.'.join(self.attrs)
    if self.extras:
        s += ' [%s]' % ','.join(self.extras)
    return s

def __repr__(self):
    return "EntryPoint.parse(%r)" % str(self)

def load(self, require=True, *args, **kwargs):
    """
    Require packages for this EntryPoint, then resolve it.
    """
    if not require or args or kwargs:
        warnings.warn(
            "Parameters to load are deprecated.  Call .resolve and "
            ".require separately.",
            PkgResourcesDeprecationWarning,
            stacklevel=2,
        )
    if require:
        self.require(*args, **kwargs)
    return self.resolve()

def resolve(self):
    """
    Resolve the entry point from its module and attrs.
    """
    module = __import__(self.module_name, fromlist=['__name__'], level=0)
    try:
        return functools.reduce(getattr, self.attrs, module)
    except AttributeError as exc:
        raise ImportError(str(exc)) from exc

def require(self, env=None, installer=None):
    if self.extras and not self.dist:
        raise UnknownExtra("Can't require() without a distribution", self)

    # Get the requirements for this entry point with all its extras and
    # then resolve them. We have to pass `extras` along when resolving so
    # that the working set knows what extras we want. Otherwise, for
    # dist-info distributions, the working set will assume that the
    # requirements for that extra are purely optional and skip over them.
    reqs = self.dist.requires(self.extras)
    items = working_set.resolve(reqs, env, installer, extras=self.extras)
    list(map(working_set.add, items))

pattern = re.compile(
    r'\s*'
    r'(?P<name>.+?)\s*'
    r'=\s*'
    r'(?P<module>[\w.]+)\s*'
    r'(:\s*(?P<attr>[\w.]+))?\s*'
    r'(?P<extras>\[.*\])?\s*$'
)

@classmethod
def parse(cls, src, dist=None):
    """Parse a single entry point from string `src`

    Entry point syntax follows the form::

        name = some.module:some.attr [extra1, extra2]

    The entry name and module name are required, but the ``:attrs`` and
    ``[extras]`` parts are optional
    """
    m = cls.pattern.match(src)
    if not m:
        msg = "EntryPoint must be in 'name=module:attrs [extras]' format"
        raise ValueError(msg, src)
    res = m.groupdict()
    extras = cls._parse_extras(res['extras'])
    attrs = res['attr'].split('.') if res['attr'] else ()
    return cls(res['name'], res['module'], attrs, extras, dist)

@classmethod
def _parse_extras(cls, extras_spec):
    if not extras_spec:
        return ()
    req = Requirement.parse('x' + extras_spec)
    if req.specs:
        raise ValueError()
    return req.extras

@classmethod
def parse_group(cls, group, lines, dist=None):
    """Parse an entry point group"""
    if not MODULE(group):
        raise ValueError("Invalid group name", group)
    this = {}
    for line in yield_lines(lines):
        ep = cls.parse(line, dist)
        if ep.name in this:
            raise ValueError("Duplicate entry point", group, ep.name)
        this[ep.name] = ep
    return this

@classmethod
def parse_map(cls, data, dist=None):
    """Parse a map of entry point groups"""
    if isinstance(data, dict):
        data = data.items()
    else:
        data = split_sections(data)
    maps = {}
    for group, lines in data:
        if group is None:
            if not lines:
                continue
            raise ValueError("Entry points must be listed in groups")
        group = group.strip()
        if group in maps:
            raise ValueError("Duplicate group name", group)
        maps[group] = cls.parse_group(group, lines, dist)
    return maps


