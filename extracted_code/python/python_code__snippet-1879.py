"""Iterator for ``FoundCandidates``.

This iterator is used when the resolver prefers to upgrade an
already-installed package. Candidates from index are returned in their
normal ordering, except replaced when the version is already installed.

The implementation iterates through and yields other candidates, inserting
the installed candidate exactly once before we start yielding older or
equivalent candidates, or after all other candidates if they are all newer.
"""
versions_found: Set[_BaseVersion] = set()
for version, func in infos:
    if version in versions_found:
        continue
    # If the installed candidate is better, yield it first.
    if installed.version >= version:
        yield installed
        versions_found.add(installed.version)
    candidate = func()
    if candidate is None:
        continue
    yield candidate
    versions_found.add(version)

# If the installed candidate is older than all other candidates.
if installed.version not in versions_found:
    yield installed


