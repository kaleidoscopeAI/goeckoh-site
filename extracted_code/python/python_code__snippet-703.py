    # For compatibility: Python before 3.9 does not support using [] on the
    # Sequence class.
    #
    # >>> from collections.abc import Sequence
    # >>> Sequence[str]
    # Traceback (most recent call last):
    #   File "<stdin>", line 1, in <module>
    # TypeError: 'ABCMeta' object is not subscriptable
    #
    # TODO: Remove this block after dropping Python 3.8 support.
    SequenceCandidate = Sequence


def _iter_built(infos: Iterator[IndexCandidateInfo]) -> Iterator[Candidate]:
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


def _iter_built_with_prepended(
    installed: Candidate, infos: Iterator[IndexCandidateInfo]
