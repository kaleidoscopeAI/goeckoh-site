def __init__(self, ws: pkg_resources.WorkingSet) -> None:
    self._ws = ws

@classmethod
def default(cls) -> BaseEnvironment:
    return cls(pkg_resources.working_set)

@classmethod
def from_paths(cls, paths: Optional[List[str]]) -> BaseEnvironment:
    return cls(pkg_resources.WorkingSet(paths))

def _iter_distributions(self) -> Iterator[BaseDistribution]:
    for dist in self._ws:
        yield Distribution(dist)

def _search_distribution(self, name: str) -> Optional[BaseDistribution]:
    """Find a distribution matching the ``name`` in the environment.

    This searches from *all* distributions available in the environment, to
    match the behavior of ``pkg_resources.get_distribution()``.
    """
    canonical_name = canonicalize_name(name)
    for dist in self.iter_all_distributions():
        if dist.canonical_name == canonical_name:
            return dist
    return None

def get_distribution(self, name: str) -> Optional[BaseDistribution]:
    # Search the distribution by looking through the working set.
    dist = self._search_distribution(name)
    if dist:
        return dist

    # If distribution could not be found, call working_set.require to
    # update the working set, and try to find the distribution again.
    # This might happen for e.g. when you install a package twice, once
    # using setup.py develop and again using setup.py install. Now when
    # running pip uninstall twice, the package gets removed from the
    # working set in the first uninstall, so we have to populate the
    # working set again so that pip knows about it and the packages gets
    # picked up and is successfully uninstalled the second time too.
    try:
        # We didn't pass in any version specifiers, so this can never
        # raise pkg_resources.VersionConflict.
        self._ws.require(name)
    except pkg_resources.DistributionNotFound:
        return None
    return self._search_distribution(name)


