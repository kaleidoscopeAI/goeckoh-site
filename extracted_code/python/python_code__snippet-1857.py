def __init__(self, candidate: Candidate) -> None:
    self.candidate = candidate

def __str__(self) -> str:
    return str(self.candidate)

def __repr__(self) -> str:
    return f"{self.__class__.__name__}({self.candidate!r})"

@property
def project_name(self) -> NormalizedName:
    # No need to canonicalize - the candidate did this
    return self.candidate.project_name

@property
def name(self) -> str:
    # No need to canonicalize - the candidate did this
    return self.candidate.name

def format_for_error(self) -> str:
    return self.candidate.format_for_error()

def get_candidate_lookup(self) -> CandidateLookup:
    return self.candidate, None

def is_satisfied_by(self, candidate: Candidate) -> bool:
    return candidate == self.candidate


