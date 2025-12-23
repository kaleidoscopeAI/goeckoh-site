def __init__(self, dist: pkg_resources.Distribution) -> None:
    self._dist = dist

@classmethod
def from_directory(cls, directory: str) -> BaseDistribution:
    dist_dir = directory.rstrip(os.sep)

    # Build a PathMetadata object, from path to metadata. :wink:
    base_dir, dist_dir_name = os.path.split(dist_dir)
    metadata = pkg_resources.PathMetadata(base_dir, dist_dir)

    # Determine the correct Distribution object type.
    if dist_dir.endswith(".egg-info"):
        dist_cls = pkg_resources.Distribution
        dist_name = os.path.splitext(dist_dir_name)[0]
    else:
        assert dist_dir.endswith(".dist-info")
        dist_cls = pkg_resources.DistInfoDistribution
        dist_name = os.path.splitext(dist_dir_name)[0].split("-")[0]

    dist = dist_cls(base_dir, project_name=dist_name, metadata=metadata)
    return cls(dist)

@classmethod
def from_metadata_file_contents(
    cls,
    metadata_contents: bytes,
    filename: str,
    project_name: str,
) -> BaseDistribution:
    metadata_dict = {
        "METADATA": metadata_contents,
    }
    dist = pkg_resources.DistInfoDistribution(
        location=filename,
        metadata=InMemoryMetadata(metadata_dict, filename),
        project_name=project_name,
    )
    return cls(dist)

@classmethod
def from_wheel(cls, wheel: Wheel, name: str) -> BaseDistribution:
    try:
        with wheel.as_zipfile() as zf:
            info_dir, _ = parse_wheel(zf, name)
            metadata_dict = {
                path.split("/", 1)[-1]: read_wheel_metadata_file(zf, path)
                for path in zf.namelist()
                if path.startswith(f"{info_dir}/")
            }
    except zipfile.BadZipFile as e:
        raise InvalidWheel(wheel.location, name) from e
    except UnsupportedWheel as e:
        raise UnsupportedWheel(f"{name} has an invalid wheel, {e}")
    dist = pkg_resources.DistInfoDistribution(
        location=wheel.location,
        metadata=InMemoryMetadata(metadata_dict, wheel.location),
        project_name=name,
    )
    return cls(dist)

@property
def location(self) -> Optional[str]:
    return self._dist.location

@property
def installed_location(self) -> Optional[str]:
    egg_link = egg_link_path_from_location(self.raw_name)
    if egg_link:
        location = egg_link
    elif self.location:
        location = self.location
    else:
        return None
    return normalize_path(location)

@property
def info_location(self) -> Optional[str]:
    return self._dist.egg_info

@property
def installed_by_distutils(self) -> bool:
    # A distutils-installed distribution is provided by FileMetadata. This
    # provider has a "path" attribute not present anywhere else. Not the
    # best introspection logic, but pip has been doing this for a long time.
    try:
        return bool(self._dist._provider.path)
    except AttributeError:
        return False

@property
def canonical_name(self) -> NormalizedName:
    return canonicalize_name(self._dist.project_name)

@property
def version(self) -> DistributionVersion:
    return parse_version(self._dist.version)

def is_file(self, path: InfoPath) -> bool:
    return self._dist.has_metadata(str(path))

def iter_distutils_script_names(self) -> Iterator[str]:
    yield from self._dist.metadata_listdir("scripts")

def read_text(self, path: InfoPath) -> str:
    name = str(path)
    if not self._dist.has_metadata(name):
        raise FileNotFoundError(name)
    content = self._dist.get_metadata(name)
    if content is None:
        raise NoneMetadataError(self, name)
    return content

def iter_entry_points(self) -> Iterable[BaseEntryPoint]:
    for group, entries in self._dist.get_entry_map().items():
        for name, entry_point in entries.items():
            name, _, value = str(entry_point).partition("=")
            yield EntryPoint(name=name.strip(), value=value.strip(), group=group)

def _metadata_impl(self) -> email.message.Message:
    """
    :raises NoneMetadataError: if the distribution reports `has_metadata()`
        True but `get_metadata()` returns None.
    """
    if isinstance(self._dist, pkg_resources.DistInfoDistribution):
        metadata_name = "METADATA"
    else:
        metadata_name = "PKG-INFO"
    try:
        metadata = self.read_text(metadata_name)
    except FileNotFoundError:
        if self.location:
            displaying_path = display_path(self.location)
        else:
            displaying_path = repr(self.location)
        logger.warning("No metadata found in %s", displaying_path)
        metadata = ""
    feed_parser = email.parser.FeedParser()
    feed_parser.feed(metadata)
    return feed_parser.close()

def iter_dependencies(self, extras: Collection[str] = ()) -> Iterable[Requirement]:
    if extras:  # pkg_resources raises on invalid extras, so we sanitize.
        extras = frozenset(pkg_resources.safe_extra(e) for e in extras)
        extras = extras.intersection(self._dist.extras)
    return self._dist.requires(extras)

def iter_provided_extras(self) -> Iterable[str]:
    return self._dist.extras

def is_extra_provided(self, extra: str) -> bool:
    return pkg_resources.safe_extra(extra) in self._dist.extras


