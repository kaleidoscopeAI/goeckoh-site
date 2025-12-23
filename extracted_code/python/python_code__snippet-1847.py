def __init__(self, paths: Sequence[str]) -> None:
    self._paths = paths

@classmethod
def default(cls) -> BaseEnvironment:
    return cls(sys.path)

@classmethod
def from_paths(cls, paths: Optional[List[str]]) -> BaseEnvironment:
    if paths is None:
        return cls(sys.path)
    return cls(paths)

def _iter_distributions(self) -> Iterator[BaseDistribution]:
    finder = _DistributionFinder()
    for location in self._paths:
        yield from finder.find(location)
        for dist in finder.find_eggs(location):
            _emit_egg_deprecation(dist.location)
            yield dist
        # This must go last because that's how pkg_resources tie-breaks.
        yield from finder.find_linked(location)

def get_distribution(self, name: str) -> Optional[BaseDistribution]:
    matches = (
        distribution
        for distribution in self.iter_all_distributions()
        if distribution.canonical_name == canonicalize_name(name)
    )
    return next(matches, None)


