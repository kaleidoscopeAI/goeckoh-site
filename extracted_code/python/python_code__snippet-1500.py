_regex = re.compile(r"^\s*" + VERSION_PATTERN + r"\s*$", re.VERBOSE | re.IGNORECASE)

def __init__(self, version: str) -> None:

    # Validate the version and parse it into pieces
    match = self._regex.search(version)
    if not match:
        raise InvalidVersion(f"Invalid version: '{version}'")

    # Store the parsed out pieces of the version
    self._version = _Version(
        epoch=int(match.group("epoch")) if match.group("epoch") else 0,
        release=tuple(int(i) for i in match.group("release").split(".")),
        pre=_parse_letter_version(match.group("pre_l"), match.group("pre_n")),
        post=_parse_letter_version(
            match.group("post_l"), match.group("post_n1") or match.group("post_n2")
        ),
        dev=_parse_letter_version(match.group("dev_l"), match.group("dev_n")),
        local=_parse_local_version(match.group("local")),
    )

    # Generate a key which will be used for sorting
    self._key = _cmpkey(
        self._version.epoch,
        self._version.release,
        self._version.pre,
        self._version.post,
        self._version.dev,
        self._version.local,
    )

def __repr__(self) -> str:
    return f"<Version('{self}')>"

def __str__(self) -> str:
    parts = []

    # Epoch
    if self.epoch != 0:
        parts.append(f"{self.epoch}!")

    # Release segment
    parts.append(".".join(str(x) for x in self.release))

    # Pre-release
    if self.pre is not None:
        parts.append("".join(str(x) for x in self.pre))

    # Post-release
    if self.post is not None:
        parts.append(f".post{self.post}")

    # Development release
    if self.dev is not None:
        parts.append(f".dev{self.dev}")

    # Local version segment
    if self.local is not None:
        parts.append(f"+{self.local}")

    return "".join(parts)

@property
def epoch(self) -> int:
    _epoch: int = self._version.epoch
    return _epoch

@property
def release(self) -> Tuple[int, ...]:
    _release: Tuple[int, ...] = self._version.release
    return _release

@property
def pre(self) -> Optional[Tuple[str, int]]:
    _pre: Optional[Tuple[str, int]] = self._version.pre
    return _pre

@property
def post(self) -> Optional[int]:
    return self._version.post[1] if self._version.post else None

@property
def dev(self) -> Optional[int]:
    return self._version.dev[1] if self._version.dev else None

@property
def local(self) -> Optional[str]:
    if self._version.local:
        return ".".join(str(x) for x in self._version.local)
    else:
        return None

@property
def public(self) -> str:
    return str(self).split("+", 1)[0]

@property
def base_version(self) -> str:
    parts = []

    # Epoch
    if self.epoch != 0:
        parts.append(f"{self.epoch}!")

    # Release segment
    parts.append(".".join(str(x) for x in self.release))

    return "".join(parts)

@property
def is_prerelease(self) -> bool:
    return self.dev is not None or self.pre is not None

@property
def is_postrelease(self) -> bool:
    return self.post is not None

@property
def is_devrelease(self) -> bool:
    return self.dev is not None

@property
def major(self) -> int:
    return self.release[0] if len(self.release) >= 1 else 0

@property
def minor(self) -> int:
    return self.release[1] if len(self.release) >= 2 else 0

@property
def micro(self) -> int:
    return self.release[2] if len(self.release) >= 3 else 0


