is_installed = True
source_link = None

def __init__(
    self,
    dist: BaseDistribution,
    template: InstallRequirement,
    factory: "Factory",
) -> None:
    self.dist = dist
    self._ireq = _make_install_req_from_dist(dist, template)
    self._factory = factory
    self._version = None

    # This is just logging some messages, so we can do it eagerly.
    # The returned dist would be exactly the same as self.dist because we
    # set satisfied_by in _make_install_req_from_dist.
    # TODO: Supply reason based on force_reinstall and upgrade_strategy.
    skip_reason = "already satisfied"
    factory.preparer.prepare_installed_requirement(self._ireq, skip_reason)

def __str__(self) -> str:
    return str(self.dist)

def __repr__(self) -> str:
    return f"{self.__class__.__name__}({self.dist!r})"

def __hash__(self) -> int:
    return hash((self.__class__, self.name, self.version))

def __eq__(self, other: Any) -> bool:
    if isinstance(other, self.__class__):
        return self.name == other.name and self.version == other.version
    return False

@property
def project_name(self) -> NormalizedName:
    return self.dist.canonical_name

@property
def name(self) -> str:
    return self.project_name

@property
def version(self) -> CandidateVersion:
    if self._version is None:
        self._version = self.dist.version
    return self._version

@property
def is_editable(self) -> bool:
    return self.dist.editable

def format_for_error(self) -> str:
    return f"{self.name} {self.version} (Installed)"

def iter_dependencies(self, with_requires: bool) -> Iterable[Optional[Requirement]]:
    if not with_requires:
        return
    for r in self.dist.iter_dependencies():
        yield from self._factory.make_requirements_from_spec(str(r), self._ireq)

def get_install_requirement(self) -> Optional[InstallRequirement]:
    return None


