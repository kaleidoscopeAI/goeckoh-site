"""Searchable snapshot of distributions on a search path"""

def __init__(
    self, search_path=None, platform=get_supported_platform(), python=PY_MAJOR
):
    """Snapshot distributions available on a search path

    Any distributions found on `search_path` are added to the environment.
    `search_path` should be a sequence of ``sys.path`` items.  If not
    supplied, ``sys.path`` is used.

    `platform` is an optional string specifying the name of the platform
    that platform-specific distributions must be compatible with.  If
    unspecified, it defaults to the current platform.  `python` is an
    optional string naming the desired version of Python (e.g. ``'3.6'``);
    it defaults to the current version.

    You may explicitly set `platform` (and/or `python`) to ``None`` if you
    wish to map *all* distributions, not just those compatible with the
    running platform or Python version.
    """
    self._distmap = {}
    self.platform = platform
    self.python = python
    self.scan(search_path)

def can_add(self, dist):
    """Is distribution `dist` acceptable for this environment?

    The distribution must match the platform and python version
    requirements specified when this environment was created, or False
    is returned.
    """
    py_compat = (
        self.python is None
        or dist.py_version is None
        or dist.py_version == self.python
    )
    return py_compat and compatible_platforms(dist.platform, self.platform)

def remove(self, dist):
    """Remove `dist` from the environment"""
    self._distmap[dist.key].remove(dist)

def scan(self, search_path=None):
    """Scan `search_path` for distributions usable in this environment

    Any distributions found are added to the environment.
    `search_path` should be a sequence of ``sys.path`` items.  If not
    supplied, ``sys.path`` is used.  Only distributions conforming to
    the platform/python version defined at initialization are added.
    """
    if search_path is None:
        search_path = sys.path

    for item in search_path:
        for dist in find_distributions(item):
            self.add(dist)

def __getitem__(self, project_name):
    """Return a newest-to-oldest list of distributions for `project_name`

    Uses case-insensitive `project_name` comparison, assuming all the
    project's distributions use their project's name converted to all
    lowercase as their key.

    """
    distribution_key = project_name.lower()
    return self._distmap.get(distribution_key, [])

def add(self, dist):
    """Add `dist` if we ``can_add()`` it and it has not already been added"""
    if self.can_add(dist) and dist.has_version():
        dists = self._distmap.setdefault(dist.key, [])
        if dist not in dists:
            dists.append(dist)
            dists.sort(key=operator.attrgetter('hashcmp'), reverse=True)

def best_match(self, req, working_set, installer=None, replace_conflicting=False):
    """Find distribution best matching `req` and usable on `working_set`

    This calls the ``find(req)`` method of the `working_set` to see if a
    suitable distribution is already active.  (This may raise
    ``VersionConflict`` if an unsuitable version of the project is already
    active in the specified `working_set`.)  If a suitable distribution
    isn't active, this method returns the newest distribution in the
    environment that meets the ``Requirement`` in `req`.  If no suitable
    distribution is found, and `installer` is supplied, then the result of
    calling the environment's ``obtain(req, installer)`` method will be
    returned.
    """
    try:
        dist = working_set.find(req)
    except VersionConflict:
        if not replace_conflicting:
            raise
        dist = None
    if dist is not None:
        return dist
    for dist in self[req.key]:
        if dist in req:
            return dist
    # try to download/install
    return self.obtain(req, installer)

def obtain(self, requirement, installer=None):
    """Obtain a distribution matching `requirement` (e.g. via download)

    Obtain a distro that matches requirement (e.g. via download).  In the
    base ``Environment`` class, this routine just returns
    ``installer(requirement)``, unless `installer` is None, in which case
    None is returned instead.  This method is a hook that allows subclasses
    to attempt other ways of obtaining a distribution before falling back
    to the `installer` argument."""
    if installer is not None:
        return installer(requirement)

def __iter__(self):
    """Yield the unique project names of the available distributions"""
    for key in self._distmap.keys():
        if self[key]:
            yield key

def __iadd__(self, other):
    """In-place addition of a distribution or environment"""
    if isinstance(other, Distribution):
        self.add(other)
    elif isinstance(other, Environment):
        for project in other:
            for dist in other[project]:
                self.add(dist)
    else:
        raise TypeError("Can't add %r to environment" % (other,))
    return self

def __add__(self, other):
    """Add an environment or distribution to an environment"""
    new = self.__class__([], platform=None, python=None)
    for env in self, other:
        new += env
    return new


