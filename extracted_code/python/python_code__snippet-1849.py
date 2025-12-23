def __init__(
    self,
    dist: importlib.metadata.Distribution,
    info_location: Optional[BasePath],
    installed_location: Optional[BasePath],
) -> None:
    self._dist = dist
    self._info_location = info_location
    self._installed_location = installed_location

@classmethod
def from_directory(cls, directory: str) -> BaseDistribution:
    info_location = pathlib.Path(directory)
    dist = importlib.metadata.Distribution.at(info_location)
    return cls(dist, info_location, info_location.parent)

@classmethod
def from_metadata_file_contents(
    cls,
    metadata_contents: bytes,
    filename: str,
    project_name: str,
) -> BaseDistribution:
    # Generate temp dir to contain the metadata file, and write the file contents.
    temp_dir = pathlib.Path(
        TempDirectory(kind="metadata", globally_managed=True).path
    )
    metadata_path = temp_dir / "METADATA"
    metadata_path.write_bytes(metadata_contents)
    # Construct dist pointing to the newly created directory.
    dist = importlib.metadata.Distribution.at(metadata_path.parent)
    return cls(dist, metadata_path.parent, None)

@classmethod
def from_wheel(cls, wheel: Wheel, name: str) -> BaseDistribution:
    try:
        with wheel.as_zipfile() as zf:
            dist = WheelDistribution.from_zipfile(zf, name, wheel.location)
    except zipfile.BadZipFile as e:
        raise InvalidWheel(wheel.location, name) from e
    except UnsupportedWheel as e:
        raise UnsupportedWheel(f"{name} has an invalid wheel, {e}")
    return cls(dist, dist.info_location, pathlib.PurePosixPath(wheel.location))

@property
def location(self) -> Optional[str]:
    if self._info_location is None:
        return None
    return str(self._info_location.parent)

@property
def info_location(self) -> Optional[str]:
    if self._info_location is None:
        return None
    return str(self._info_location)

@property
def installed_location(self) -> Optional[str]:
    if self._installed_location is None:
        return None
    return normalize_path(str(self._installed_location))

def _get_dist_name_from_location(self) -> Optional[str]:
    """Try to get the name from the metadata directory name.

    This is much faster than reading metadata.
    """
    if self._info_location is None:
        return None
    stem, suffix = os.path.splitext(self._info_location.name)
    if suffix not in (".dist-info", ".egg-info"):
        return None
    return stem.split("-", 1)[0]

@property
def canonical_name(self) -> NormalizedName:
    name = self._get_dist_name_from_location() or get_dist_name(self._dist)
    return canonicalize_name(name)

@property
def version(self) -> DistributionVersion:
    return parse_version(self._dist.version)

def is_file(self, path: InfoPath) -> bool:
    return self._dist.read_text(str(path)) is not None

def iter_distutils_script_names(self) -> Iterator[str]:
    # A distutils installation is always "flat" (not in e.g. egg form), so
    # if this distribution's info location is NOT a pathlib.Path (but e.g.
    # zipfile.Path), it can never contain any distutils scripts.
    if not isinstance(self._info_location, pathlib.Path):
        return
    for child in self._info_location.joinpath("scripts").iterdir():
        yield child.name

def read_text(self, path: InfoPath) -> str:
    content = self._dist.read_text(str(path))
    if content is None:
        raise FileNotFoundError(path)
    return content

def iter_entry_points(self) -> Iterable[BaseEntryPoint]:
    # importlib.metadata's EntryPoint structure sasitfies BaseEntryPoint.
    return self._dist.entry_points

def _metadata_impl(self) -> email.message.Message:
    # From Python 3.10+, importlib.metadata declares PackageMetadata as the
    # return type. This protocol is unfortunately a disaster now and misses
    # a ton of fields that we need, including get() and get_payload(). We
    # rely on the implementation that the object is actually a Message now,
    # until upstream can improve the protocol. (python/cpython#94952)
    return cast(email.message.Message, self._dist.metadata)

def iter_provided_extras(self) -> Iterable[str]:
    return self.metadata.get_all("Provides-Extra", [])

def is_extra_provided(self, extra: str) -> bool:
    return any(
        canonicalize_name(provided_extra) == canonicalize_name(extra)
        for provided_extra in self.metadata.get_all("Provides-Extra", [])
    )

def iter_dependencies(self, extras: Collection[str] = ()) -> Iterable[Requirement]:
    contexts: Sequence[Dict[str, str]] = [{"extra": e} for e in extras]
    for req_string in self.metadata.get_all("Requires-Dist", []):
        req = Requirement(req_string)
        if not req.marker:
            yield req
        elif not extras and req.marker.evaluate({"extra": ""}):
            yield req
        elif any(req.marker.evaluate(context) for context in contexts):
            yield req


