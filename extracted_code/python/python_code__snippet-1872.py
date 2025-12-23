@property
def project_name(self) -> NormalizedName:
    """The "project name" of a requirement.

    This is different from ``name`` if this requirement contains extras,
    in which case ``name`` would contain the ``[...]`` part, while this
    refers to the name of the project.
    """
    raise NotImplementedError("Subclass should override")

@property
def name(self) -> str:
    """The name identifying this requirement in the resolver.

    This is different from ``project_name`` if this requirement contains
    extras, where ``project_name`` would not contain the ``[...]`` part.
    """
    raise NotImplementedError("Subclass should override")

def is_satisfied_by(self, candidate: "Candidate") -> bool:
    return False

def get_candidate_lookup(self) -> CandidateLookup:
    raise NotImplementedError("Subclass should override")

def format_for_error(self) -> str:
    raise NotImplementedError("Subclass should override")


