"""A requirement that cannot be satisfied."""

def __init__(self, name: NormalizedName) -> None:
    self._name = name

def __str__(self) -> str:
    return f"{self._name} (unavailable)"

def __repr__(self) -> str:
    return f"{self.__class__.__name__}({str(self._name)!r})"

@property
def project_name(self) -> NormalizedName:
    return self._name

@property
def name(self) -> str:
    return self._name

def format_for_error(self) -> str:
    return str(self)

def get_candidate_lookup(self) -> CandidateLookup:
    return None, None

def is_satisfied_by(self, candidate: Candidate) -> bool:
    return False


