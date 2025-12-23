def __init__(self, ireq: InstallRequirement) -> None:
    assert ireq.link is None, "This is a link, not a specifier"
    self._ireq = ireq
    self._extras = frozenset(canonicalize_name(e) for e in self._ireq.extras)

def __str__(self) -> str:
    return str(self._ireq.req)

def __repr__(self) -> str:
    return f"{self.__class__.__name__}({str(self._ireq.req)!r})"

@property
def project_name(self) -> NormalizedName:
    assert self._ireq.req, "Specifier-backed ireq is always PEP 508"
    return canonicalize_name(self._ireq.req.name)

@property
def name(self) -> str:
    return format_name(self.project_name, self._extras)

def format_for_error(self) -> str:
    # Convert comma-separated specifiers into "A, B, ..., F and G"
    # This makes the specifier a bit more "human readable", without
    # risking a change in meaning. (Hopefully! Not all edge cases have
    # been checked)
    parts = [s.strip() for s in str(self).split(",")]
    if len(parts) == 0:
        return ""
    elif len(parts) == 1:
        return parts[0]

    return ", ".join(parts[:-1]) + " and " + parts[-1]

def get_candidate_lookup(self) -> CandidateLookup:
    return None, self._ireq

def is_satisfied_by(self, candidate: Candidate) -> bool:
    assert candidate.name == self.name, (
        f"Internal issue: Candidate is not for this requirement "
        f"{candidate.name} vs {self.name}"
    )
    # We can safely always allow prereleases here since PackageFinder
    # already implements the prerelease logic, and would have filtered out
    # prerelease candidates if the user does not expect them.
    assert self._ireq.req, "Specifier-backed ireq is always PEP 508"
    spec = self._ireq.req.specifier
    return spec.contains(candidate.version, prereleases=True)


