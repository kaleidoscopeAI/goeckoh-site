    from typing import Protocol

    class ConflictCause(Protocol):
        requirement: RequiresPythonRequirement
        parent: Candidate


