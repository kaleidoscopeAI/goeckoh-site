is_installed = False
source_link = None

def __init__(self, py_version_info: Optional[Tuple[int, ...]]) -> None:
    if py_version_info is not None:
        version_info = normalize_version_info(py_version_info)
    else:
        version_info = sys.version_info[:3]
    self._version = Version(".".join(str(c) for c in version_info))

# We don't need to implement __eq__() and __ne__() since there is always
# only one RequiresPythonCandidate in a resolution, i.e. the host Python.
# The built-in object.__eq__() and object.__ne__() do exactly what we want.

def __str__(self) -> str:
    return f"Python {self._version}"

@property
def project_name(self) -> NormalizedName:
    return REQUIRES_PYTHON_IDENTIFIER

@property
def name(self) -> str:
    return REQUIRES_PYTHON_IDENTIFIER

@property
def version(self) -> CandidateVersion:
    return self._version

def format_for_error(self) -> str:
    return f"Python {self.version}"

def iter_dependencies(self, with_requires: bool) -> Iterable[Optional[Requirement]]:
    return ()

def get_install_requirement(self) -> Optional[InstallRequirement]:
    return None


