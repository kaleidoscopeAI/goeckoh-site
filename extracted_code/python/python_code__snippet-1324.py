"""
A base class for distributions, whether installed or from indexes.
Either way, it must have some metadata, so that's all that's needed
for construction.
"""

build_time_dependency = False
"""
Set to True if it's known to be only a build-time dependency (i.e.
not needed after installation).
"""

requested = False
"""A boolean that indicates whether the ``REQUESTED`` metadata file is
present (in other words, whether the package was installed by user
request or it was installed as a dependency)."""

def __init__(self, metadata):
    """
    Initialise an instance.
    :param metadata: The instance of :class:`Metadata` describing this
    distribution.
    """
    self.metadata = metadata
    self.name = metadata.name
    self.key = self.name.lower()  # for case-insensitive comparisons
    self.version = metadata.version
    self.locator = None
    self.digest = None
    self.extras = None  # additional features requested
    self.context = None  # environment marker overrides
    self.download_urls = set()
    self.digests = {}

@property
def source_url(self):
    """
    The source archive download URL for this distribution.
    """
    return self.metadata.source_url

download_url = source_url  # Backward compatibility

@property
def name_and_version(self):
    """
    A utility property which displays the name and version in parentheses.
    """
    return '%s (%s)' % (self.name, self.version)

@property
def provides(self):
    """
    A set of distribution names and versions provided by this distribution.
    :return: A set of "name (version)" strings.
    """
    plist = self.metadata.provides
    s = '%s (%s)' % (self.name, self.version)
    if s not in plist:
        plist.append(s)
    return plist

def _get_requirements(self, req_attr):
    md = self.metadata
    reqts = getattr(md, req_attr)
    logger.debug('%s: got requirements %r from metadata: %r', self.name,
                 req_attr, reqts)
    return set(
        md.get_requirements(reqts, extras=self.extras, env=self.context))

@property
def run_requires(self):
    return self._get_requirements('run_requires')

@property
def meta_requires(self):
    return self._get_requirements('meta_requires')

@property
def build_requires(self):
    return self._get_requirements('build_requires')

@property
def test_requires(self):
    return self._get_requirements('test_requires')

@property
def dev_requires(self):
    return self._get_requirements('dev_requires')

def matches_requirement(self, req):
    """
    Say if this instance matches (fulfills) a requirement.
    :param req: The requirement to match.
    :rtype req: str
    :return: True if it matches, else False.
    """
    # Requirement may contain extras - parse to lose those
    # from what's passed to the matcher
    r = parse_requirement(req)
    scheme = get_scheme(self.metadata.scheme)
    try:
        matcher = scheme.matcher(r.requirement)
    except UnsupportedVersionError:
        # XXX compat-mode if cannot read the version
        logger.warning('could not read version %r - using name only', req)
        name = req.split()[0]
        matcher = scheme.matcher(name)

    name = matcher.key  # case-insensitive

    result = False
    for p in self.provides:
        p_name, p_ver = parse_name_and_version(p)
        if p_name != name:
            continue
        try:
            result = matcher.match(p_ver)
            break
        except UnsupportedVersionError:
            pass
    return result

def __repr__(self):
    """
    Return a textual representation of this instance,
    """
    if self.source_url:
        suffix = ' [%s]' % self.source_url
    else:
        suffix = ''
    return '<Distribution %s (%s)%s>' % (self.name, self.version, suffix)

def __eq__(self, other):
    """
    See if this distribution is the same as another.
    :param other: The distribution to compare with. To be equal to one
                  another. distributions must have the same type, name,
                  version and source_url.
    :return: True if it is the same, else False.
    """
    if type(other) is not type(self):
        result = False
    else:
        result = (self.name == other.name and self.version == other.version
                  and self.source_url == other.source_url)
    return result

def __hash__(self):
    """
    Compute hash in a way which matches the equality test.
    """
    return hash(self.name) + hash(self.version) + hash(self.source_url)


