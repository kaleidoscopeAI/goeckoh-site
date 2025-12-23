"""Iterator for ``FoundCandidates``.

This iterator is used when the resolver prefers the already-installed
candidate and NOT to upgrade. The installed candidate is therefore
always yielded first, and candidates from index come later in their
normal ordering, except skipped when the version is already installed.
"""
yield installed
versions_found: Set[_BaseVersion] = {installed.version}
for version, func in infos:
    if version in versions_found:
        continue
    candidate = func()
    if candidate is None:
        continue
    yield candidate
    versions_found.add(version)


