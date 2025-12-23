"""Iterator for ``FoundCandidates``.

This iterator is used when the package is not already installed. Candidates
from index come later in their normal ordering.
"""
versions_found: Set[_BaseVersion] = set()
for version, func in infos:
    if version in versions_found:
        continue
    candidate = func()
    if candidate is None:
        continue
    yield candidate
    versions_found.add(version)


