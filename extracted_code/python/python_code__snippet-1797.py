"""Wrap an actual or potential sys.path entry w/metadata"""

PKG_INFO = 'PKG-INFO'

def __init__(
    self,
    location=None,
    metadata=None,
    project_name=None,
    version=None,
    py_version=PY_MAJOR,
    platform=None,
    precedence=EGG_DIST,
):
    self.project_name = safe_name(project_name or 'Unknown')
    if version is not None:
        self._version = safe_version(version)
    self.py_version = py_version
    self.platform = platform
    self.location = location
    self.precedence = precedence
    self._provider = metadata or empty_provider

@classmethod
def from_location(cls, location, basename, metadata=None, **kw):
    project_name, version, py_version, platform = [None] * 4
    basename, ext = os.path.splitext(basename)
    if ext.lower() in _distributionImpl:
        cls = _distributionImpl[ext.lower()]

        match = EGG_NAME(basename)
        if match:
            project_name, version, py_version, platform = match.group(
                'name', 'ver', 'pyver', 'plat'
            )
    return cls(
        location,
        metadata,
        project_name=project_name,
        version=version,
        py_version=py_version,
        platform=platform,
        **kw,
    )._reload_version()

def _reload_version(self):
    return self

@property
def hashcmp(self):
    return (
        self._forgiving_parsed_version,
        self.precedence,
        self.key,
        self.location,
        self.py_version or '',
        self.platform or '',
    )

def __hash__(self):
    return hash(self.hashcmp)

def __lt__(self, other):
    return self.hashcmp < other.hashcmp

def __le__(self, other):
    return self.hashcmp <= other.hashcmp

def __gt__(self, other):
    return self.hashcmp > other.hashcmp

def __ge__(self, other):
    return self.hashcmp >= other.hashcmp

def __eq__(self, other):
    if not isinstance(other, self.__class__):
        # It's not a Distribution, so they are not equal
        return False
    return self.hashcmp == other.hashcmp

def __ne__(self, other):
    return not self == other

# These properties have to be lazy so that we don't have to load any
# metadata until/unless it's actually needed.  (i.e., some distributions
# may not know their name or version without loading PKG-INFO)

@property
def key(self):
    try:
        return self._key
    except AttributeError:
        self._key = key = self.project_name.lower()
        return key

@property
def parsed_version(self):
    if not hasattr(self, "_parsed_version"):
        try:
            self._parsed_version = parse_version(self.version)
        except packaging.version.InvalidVersion as ex:
            info = f"(package: {self.project_name})"
            if hasattr(ex, "add_note"):
                ex.add_note(info)  # PEP 678
                raise
            raise packaging.version.InvalidVersion(f"{str(ex)} {info}") from None

    return self._parsed_version

@property
def _forgiving_parsed_version(self):
    try:
        return self.parsed_version
    except packaging.version.InvalidVersion as ex:
        self._parsed_version = parse_version(_forgiving_version(self.version))

        notes = "\n".join(getattr(ex, "__notes__", []))  # PEP 678
        msg = f"""!!\n\n
        *************************************************************************
        {str(ex)}\n{notes}

        This is a long overdue deprecation.
        For the time being, `pkg_resources` will use `{self._parsed_version}`
        as a replacement to avoid breaking existing environments,
        but no future compatibility is guaranteed.

        If you maintain package {self.project_name} you should implement
        the relevant changes to adequate the project to PEP 440 immediately.
        *************************************************************************
        \n\n!!
        """
        warnings.warn(msg, DeprecationWarning)

        return self._parsed_version

@property
def version(self):
    try:
        return self._version
    except AttributeError as e:
        version = self._get_version()
        if version is None:
            path = self._get_metadata_path_for_display(self.PKG_INFO)
            msg = ("Missing 'Version:' header and/or {} file at path: {}").format(
                self.PKG_INFO, path
            )
            raise ValueError(msg, self) from e

        return version

@property
def _dep_map(self):
    """
    A map of extra to its list of (direct) requirements
    for this distribution, including the null extra.
    """
    try:
        return self.__dep_map
    except AttributeError:
        self.__dep_map = self._filter_extras(self._build_dep_map())
    return self.__dep_map

@staticmethod
def _filter_extras(dm):
    """
    Given a mapping of extras to dependencies, strip off
    environment markers and filter out any dependencies
    not matching the markers.
    """
    for extra in list(filter(None, dm)):
        new_extra = extra
        reqs = dm.pop(extra)
        new_extra, _, marker = extra.partition(':')
        fails_marker = marker and (
            invalid_marker(marker) or not evaluate_marker(marker)
        )
        if fails_marker:
            reqs = []
        new_extra = safe_extra(new_extra) or None

        dm.setdefault(new_extra, []).extend(reqs)
    return dm

def _build_dep_map(self):
    dm = {}
    for name in 'requires.txt', 'depends.txt':
        for extra, reqs in split_sections(self._get_metadata(name)):
            dm.setdefault(extra, []).extend(parse_requirements(reqs))
    return dm

def requires(self, extras=()):
    """List of Requirements needed for this distro if `extras` are used"""
    dm = self._dep_map
    deps = []
    deps.extend(dm.get(None, ()))
    for ext in extras:
        try:
            deps.extend(dm[safe_extra(ext)])
        except KeyError as e:
            raise UnknownExtra(
                "%s has no such extra feature %r" % (self, ext)
            ) from e
    return deps

def _get_metadata_path_for_display(self, name):
    """
    Return the path to the given metadata file, if available.
    """
    try:
        # We need to access _get_metadata_path() on the provider object
        # directly rather than through this class's __getattr__()
        # since _get_metadata_path() is marked private.
        path = self._provider._get_metadata_path(name)

    # Handle exceptions e.g. in case the distribution's metadata
    # provider doesn't support _get_metadata_path().
    except Exception:
        return '[could not detect]'

    return path

def _get_metadata(self, name):
    if self.has_metadata(name):
        for line in self.get_metadata_lines(name):
            yield line

def _get_version(self):
    lines = self._get_metadata(self.PKG_INFO)
    version = _version_from_file(lines)

    return version

def activate(self, path=None, replace=False):
    """Ensure distribution is importable on `path` (default=sys.path)"""
    if path is None:
        path = sys.path
    self.insert_on(path, replace=replace)
    if path is sys.path:
        fixup_namespace_packages(self.location)
        for pkg in self._get_metadata('namespace_packages.txt'):
            if pkg in sys.modules:
                declare_namespace(pkg)

def egg_name(self):
    """Return what this distribution's standard .egg filename should be"""
    filename = "%s-%s-py%s" % (
        to_filename(self.project_name),
        to_filename(self.version),
        self.py_version or PY_MAJOR,
    )

    if self.platform:
        filename += '-' + self.platform
    return filename

def __repr__(self):
    if self.location:
        return "%s (%s)" % (self, self.location)
    else:
        return str(self)

def __str__(self):
    try:
        version = getattr(self, 'version', None)
    except ValueError:
        version = None
    version = version or "[unknown version]"
    return "%s %s" % (self.project_name, version)

def __getattr__(self, attr):
    """Delegate all unrecognized public attributes to .metadata provider"""
    if attr.startswith('_'):
        raise AttributeError(attr)
    return getattr(self._provider, attr)

def __dir__(self):
    return list(
        set(super(Distribution, self).__dir__())
        | set(attr for attr in self._provider.__dir__() if not attr.startswith('_'))
    )

@classmethod
def from_filename(cls, filename, metadata=None, **kw):
    return cls.from_location(
        _normalize_cached(filename), os.path.basename(filename), metadata, **kw
    )

def as_requirement(self):
    """Return a ``Requirement`` that matches this distribution exactly"""
    if isinstance(self.parsed_version, packaging.version.Version):
        spec = "%s==%s" % (self.project_name, self.parsed_version)
    else:
        spec = "%s===%s" % (self.project_name, self.parsed_version)

    return Requirement.parse(spec)

def load_entry_point(self, group, name):
    """Return the `name` entry point of `group` or raise ImportError"""
    ep = self.get_entry_info(group, name)
    if ep is None:
        raise ImportError("Entry point %r not found" % ((group, name),))
    return ep.load()

def get_entry_map(self, group=None):
    """Return the entry point map for `group`, or the full entry map"""
    try:
        ep_map = self._ep_map
    except AttributeError:
        ep_map = self._ep_map = EntryPoint.parse_map(
            self._get_metadata('entry_points.txt'), self
        )
    if group is not None:
        return ep_map.get(group, {})
    return ep_map

def get_entry_info(self, group, name):
    """Return the EntryPoint object for `group`+`name`, or ``None``"""
    return self.get_entry_map(group).get(name)

# FIXME: 'Distribution.insert_on' is too complex (13)
def insert_on(self, path, loc=None, replace=False):  # noqa: C901
    """Ensure self.location is on path

    If replace=False (default):
        - If location is already in path anywhere, do nothing.
        - Else:
          - If it's an egg and its parent directory is on path,
            insert just ahead of the parent.
          - Else: add to the end of path.
    If replace=True:
        - If location is already on path anywhere (not eggs)
          or higher priority than its parent (eggs)
          do nothing.
        - Else:
          - If it's an egg and its parent directory is on path,
            insert just ahead of the parent,
            removing any lower-priority entries.
          - Else: add it to the front of path.
    """

    loc = loc or self.location
    if not loc:
        return

    nloc = _normalize_cached(loc)
    bdir = os.path.dirname(nloc)
    npath = [(p and _normalize_cached(p) or p) for p in path]

    for p, item in enumerate(npath):
        if item == nloc:
            if replace:
                break
            else:
                # don't modify path (even removing duplicates) if
                # found and not replace
                return
        elif item == bdir and self.precedence == EGG_DIST:
            # if it's an .egg, give it precedence over its directory
            # UNLESS it's already been added to sys.path and replace=False
            if (not replace) and nloc in npath[p:]:
                return
            if path is sys.path:
                self.check_version_conflict()
            path.insert(p, loc)
            npath.insert(p, nloc)
            break
    else:
        if path is sys.path:
            self.check_version_conflict()
        if replace:
            path.insert(0, loc)
        else:
            path.append(loc)
        return

    # p is the spot where we found or inserted loc; now remove duplicates
    while True:
        try:
            np = npath.index(nloc, p + 1)
        except ValueError:
            break
        else:
            del npath[np], path[np]
            # ha!
            p = np

    return

def check_version_conflict(self):
    if self.key == 'setuptools':
        # ignore the inevitable setuptools self-conflicts  :(
        return

    nsp = dict.fromkeys(self._get_metadata('namespace_packages.txt'))
    loc = normalize_path(self.location)
    for modname in self._get_metadata('top_level.txt'):
        if (
            modname not in sys.modules
            or modname in nsp
            or modname in _namespace_packages
        ):
            continue
        if modname in ('pkg_resources', 'setuptools', 'site'):
            continue
        fn = getattr(sys.modules[modname], '__file__', None)
        if fn and (
            normalize_path(fn).startswith(loc) or fn.startswith(self.location)
        ):
            continue
        issue_warning(
            "Module %s was already imported from %s, but %s is being added"
            " to sys.path" % (modname, fn, self.location),
        )

def has_version(self):
    try:
        self.version
    except ValueError:
        issue_warning("Unbuilt egg for " + repr(self))
        return False
    except SystemError:
        # TODO: remove this except clause when python/cpython#103632 is fixed.
        return False
    return True

def clone(self, **kw):
    """Copy this distribution, substituting in any changed keyword args"""
    names = 'project_name version py_version platform location precedence'
    for attr in names.split():
        kw.setdefault(attr, getattr(self, attr, None))
    kw.setdefault('metadata', self._provider)
    return self.__class__(**kw)

@property
def extras(self):
    return [dep for dep in self._dep_map if dep]


