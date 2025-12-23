"""A workalike for Hashes used when we're missing a hash for a requirement

It computes the actual hash of the requirement and raises a HashMissing
exception showing it to the user.

"""

def __init__(self) -> None:
    """Don't offer the ``hashes`` kwarg."""
    # Pass our favorite hash in to generate a "gotten hash". With the
    # empty list, it will never match, so an error will always raise.
    super().__init__(hashes={FAVORITE_HASH: []})

def _raise(self, gots: Dict[str, "_Hash"]) -> "NoReturn":
    raise HashMissing(gots[FAVORITE_HASH].hexdigest())


