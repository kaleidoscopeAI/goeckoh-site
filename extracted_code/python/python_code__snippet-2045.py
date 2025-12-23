"""Links to content may have embedded hash values. This class parses those.

`name` must be any member of `_SUPPORTED_HASHES`.

This class can be converted to and from `ArchiveInfo`. While ArchiveInfo intends to
be JSON-serializable to conform to PEP 610, this class contains the logic for
parsing a hash name and value for correctness, and then checking whether that hash
conforms to a schema with `.is_hash_allowed()`."""

name: str
value: str

_hash_url_fragment_re = re.compile(
    # NB: we do not validate that the second group (.*) is a valid hex
    # digest. Instead, we simply keep that string in this class, and then check it
    # against Hashes when hash-checking is needed. This is easier to debug than
    # proactively discarding an invalid hex digest, as we handle incorrect hashes
    # and malformed hashes in the same place.
    r"[#&]({choices})=([^&]*)".format(
        choices="|".join(re.escape(hash_name) for hash_name in _SUPPORTED_HASHES)
    ),
)

def __post_init__(self) -> None:
    assert self.name in _SUPPORTED_HASHES

@classmethod
@functools.lru_cache(maxsize=None)
def find_hash_url_fragment(cls, url: str) -> Optional["LinkHash"]:
    """Search a string for a checksum algorithm name and encoded output value."""
    match = cls._hash_url_fragment_re.search(url)
    if match is None:
        return None
    name, value = match.groups()
    return cls(name=name, value=value)

def as_dict(self) -> Dict[str, str]:
    return {self.name: self.value}

def as_hashes(self) -> Hashes:
    """Return a Hashes instance which checks only for the current hash."""
    return Hashes({self.name: [self.value]})

def is_hash_allowed(self, hashes: Optional[Hashes]) -> bool:
    """
    Return True if the current hash is allowed by `hashes`.
    """
    if hashes is None:
        return False
    return hashes.is_hash_allowed(self.name, hex_digest=self.value)


