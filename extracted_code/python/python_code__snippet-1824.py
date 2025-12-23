"""A hash was needed for a requirement but is absent."""

order = 2
head = (
    "Hashes are required in --require-hashes mode, but they are "
    "missing from some requirements. Here is a list of those "
    "requirements along with the hashes their downloaded archives "
    "actually had. Add lines like these to your requirements files to "
    "prevent tampering. (If you did not enable --require-hashes "
    "manually, note that it turns on automatically when any package "
    "has a hash.)"
)

def __init__(self, gotten_hash: str) -> None:
    """
    :param gotten_hash: The hash of the (possibly malicious) archive we
        just downloaded
    """
    self.gotten_hash = gotten_hash

def body(self) -> str:
    # Dodge circular import.
    from pip._internal.utils.hashes import FAVORITE_HASH

    package = None
    if self.req:
        # In the case of URL-based requirements, display the original URL
        # seen in the requirements file rather than the package name,
        # so the output can be directly copied into the requirements file.
        package = (
            self.req.original_link
            if self.req.is_direct
            # In case someone feeds something downright stupid
            # to InstallRequirement's constructor.
            else getattr(self.req, "req", None)
        )
    return "    {} --hash={}:{}".format(
        package or "unknown package", FAVORITE_HASH, self.gotten_hash
    )


