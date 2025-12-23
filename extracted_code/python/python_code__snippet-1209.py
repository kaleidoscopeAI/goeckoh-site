# Avoid conflicting with the PyPI package "Python".
REQUIRES_PYTHON_IDENTIFIER = cast(NormalizedName, "<Python from Requires-Python>")


def as_base_candidate(candidate: Candidate) -> Optional[BaseCandidate]:
    """The runtime version of BaseCandidate."""
    base_candidate_classes = (
        AlreadyInstalledCandidate,
        EditableCandidate,
        LinkCandidate,
    )
    if isinstance(candidate, base_candidate_classes):
        return candidate
    return None


def make_install_req_from_link(
    link: Link, template: InstallRequirement
