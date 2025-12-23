"""
Created with the *path* of the ``.dist-info`` directory provided to the
constructor. It reads the metadata contained in ``pydist.json`` when it is
instantiated., or uses a passed in Metadata instance (useful for when
dry-run mode is being used).
"""

hasher = 'sha256'

def __init__(self, path, metadata=None, env=None):
    self.modules = []
    self.finder = finder = resources.finder_for_path(path)
    if finder is None:
        raise ValueError('finder unavailable for %s' % path)
    if env and env._cache_enabled and path in env._cache.path:
        metadata = env._cache.path[path].metadata
    elif metadata is None:
        r = finder.find(METADATA_FILENAME)
        # Temporary - for Wheel 0.23 support
        if r is None:
            r = finder.find(WHEEL_METADATA_FILENAME)
        # Temporary - for legacy support
        if r is None:
            r = finder.find(LEGACY_METADATA_FILENAME)
        if r is None:
            raise ValueError('no %s found in %s' %
                             (METADATA_FILENAME, path))
        with contextlib.closing(r.as_stream()) as stream:
            metadata = Metadata(fileobj=stream, scheme='legacy')

    super(InstalledDistribution, self).__init__(metadata, path, env)

    if env and env._cache_enabled:
        env._cache.add(self)

    r = finder.find('REQUESTED')
    self.requested = r is not None
    p = os.path.join(path, 'top_level.txt')
    if os.path.exists(p):
        with open(p, 'rb') as f:
            data = f.read().decode('utf-8')
        self.modules = data.splitlines()

def __repr__(self):
    return '<InstalledDistribution %r %s at %r>' % (
        self.name, self.version, self.path)

def __str__(self):
    return "%s %s" % (self.name, self.version)

def _get_records(self):
    """
    Get the list of installed files for the distribution
    :return: A list of tuples of path, hash and size. Note that hash and
             size might be ``None`` for some entries. The path is exactly
             as stored in the file (which is as in PEP 376).
    """
    results = []
    r = self.get_distinfo_resource('RECORD')
    with contextlib.closing(r.as_stream()) as stream:
        with CSVReader(stream=stream) as record_reader:
            # Base location is parent dir of .dist-info dir
            # base_location = os.path.dirname(self.path)
            # base_location = os.path.abspath(base_location)
            for row in record_reader:
                missing = [None for i in range(len(row), 3)]
                path, checksum, size = row + missing
                # if not os.path.isabs(path):
                #     path = path.replace('/', os.sep)
                #     path = os.path.join(base_location, path)
                results.append((path, checksum, size))
    return results

@cached_property
def exports(self):
    """
    Return the information exported by this distribution.
    :return: A dictionary of exports, mapping an export category to a dict
             of :class:`ExportEntry` instances describing the individual
             export entries, and keyed by name.
    """
    result = {}
    r = self.get_distinfo_resource(EXPORTS_FILENAME)
    if r:
        result = self.read_exports()
    return result

def read_exports(self):
    """
    Read exports data from a file in .ini format.

    :return: A dictionary of exports, mapping an export category to a list
             of :class:`ExportEntry` instances describing the individual
             export entries.
    """
    result = {}
    r = self.get_distinfo_resource(EXPORTS_FILENAME)
    if r:
        with contextlib.closing(r.as_stream()) as stream:
            result = read_exports(stream)
    return result

def write_exports(self, exports):
    """
    Write a dictionary of exports to a file in .ini format.
    :param exports: A dictionary of exports, mapping an export category to
                    a list of :class:`ExportEntry` instances describing the
                    individual export entries.
    """
    rf = self.get_distinfo_file(EXPORTS_FILENAME)
    with open(rf, 'w') as f:
        write_exports(exports, f)

def get_resource_path(self, relative_path):
    """
    NOTE: This API may change in the future.

    Return the absolute path to a resource file with the given relative
    path.

    :param relative_path: The path, relative to .dist-info, of the resource
                          of interest.
    :return: The absolute path where the resource is to be found.
    """
    r = self.get_distinfo_resource('RESOURCES')
    with contextlib.closing(r.as_stream()) as stream:
        with CSVReader(stream=stream) as resources_reader:
            for relative, destination in resources_reader:
                if relative == relative_path:
                    return destination
    raise KeyError('no resource file with relative path %r '
                   'is installed' % relative_path)

def list_installed_files(self):
    """
    Iterates over the ``RECORD`` entries and returns a tuple
    ``(path, hash, size)`` for each line.

    :returns: iterator of (path, hash, size)
    """
    for result in self._get_records():
        yield result

def write_installed_files(self, paths, prefix, dry_run=False):
    """
    Writes the ``RECORD`` file, using the ``paths`` iterable passed in. Any
    existing ``RECORD`` file is silently overwritten.

    prefix is used to determine when to write absolute paths.
    """
    prefix = os.path.join(prefix, '')
    base = os.path.dirname(self.path)
    base_under_prefix = base.startswith(prefix)
    base = os.path.join(base, '')
    record_path = self.get_distinfo_file('RECORD')
    logger.info('creating %s', record_path)
    if dry_run:
        return None
    with CSVWriter(record_path) as writer:
        for path in paths:
            if os.path.isdir(path) or path.endswith(('.pyc', '.pyo')):
                # do not put size and hash, as in PEP-376
                hash_value = size = ''
            else:
                size = '%d' % os.path.getsize(path)
                with open(path, 'rb') as fp:
                    hash_value = self.get_hash(fp.read())
            if path.startswith(base) or (base_under_prefix
                                         and path.startswith(prefix)):
                path = os.path.relpath(path, base)
            writer.writerow((path, hash_value, size))

        # add the RECORD file itself
        if record_path.startswith(base):
            record_path = os.path.relpath(record_path, base)
        writer.writerow((record_path, '', ''))
    return record_path

def check_installed_files(self):
    """
    Checks that the hashes and sizes of the files in ``RECORD`` are
    matched by the files themselves. Returns a (possibly empty) list of
    mismatches. Each entry in the mismatch list will be a tuple consisting
    of the path, 'exists', 'size' or 'hash' according to what didn't match
    (existence is checked first, then size, then hash), the expected
    value and the actual value.
    """
    mismatches = []
    base = os.path.dirname(self.path)
    record_path = self.get_distinfo_file('RECORD')
    for path, hash_value, size in self.list_installed_files():
        if not os.path.isabs(path):
            path = os.path.join(base, path)
        if path == record_path:
            continue
        if not os.path.exists(path):
            mismatches.append((path, 'exists', True, False))
        elif os.path.isfile(path):
            actual_size = str(os.path.getsize(path))
            if size and actual_size != size:
                mismatches.append((path, 'size', size, actual_size))
            elif hash_value:
                if '=' in hash_value:
                    hasher = hash_value.split('=', 1)[0]
                else:
                    hasher = None

                with open(path, 'rb') as f:
                    actual_hash = self.get_hash(f.read(), hasher)
                    if actual_hash != hash_value:
                        mismatches.append(
                            (path, 'hash', hash_value, actual_hash))
    return mismatches

@cached_property
def shared_locations(self):
    """
    A dictionary of shared locations whose keys are in the set 'prefix',
    'purelib', 'platlib', 'scripts', 'headers', 'data' and 'namespace'.
    The corresponding value is the absolute path of that category for
    this distribution, and takes into account any paths selected by the
    user at installation time (e.g. via command-line arguments). In the
    case of the 'namespace' key, this would be a list of absolute paths
    for the roots of namespace packages in this distribution.

    The first time this property is accessed, the relevant information is
    read from the SHARED file in the .dist-info directory.
    """
    result = {}
    shared_path = os.path.join(self.path, 'SHARED')
    if os.path.isfile(shared_path):
        with codecs.open(shared_path, 'r', encoding='utf-8') as f:
            lines = f.read().splitlines()
        for line in lines:
            key, value = line.split('=', 1)
            if key == 'namespace':
                result.setdefault(key, []).append(value)
            else:
                result[key] = value
    return result

def write_shared_locations(self, paths, dry_run=False):
    """
    Write shared location information to the SHARED file in .dist-info.
    :param paths: A dictionary as described in the documentation for
    :meth:`shared_locations`.
    :param dry_run: If True, the action is logged but no file is actually
                    written.
    :return: The path of the file written to.
    """
    shared_path = os.path.join(self.path, 'SHARED')
    logger.info('creating %s', shared_path)
    if dry_run:
        return None
    lines = []
    for key in ('prefix', 'lib', 'headers', 'scripts', 'data'):
        path = paths[key]
        if os.path.isdir(paths[key]):
            lines.append('%s=%s' % (key, path))
    for ns in paths.get('namespace', ()):
        lines.append('namespace=%s' % ns)

    with codecs.open(shared_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    return shared_path

def get_distinfo_resource(self, path):
    if path not in DIST_FILES:
        raise DistlibException('invalid path for a dist-info file: '
                               '%r at %r' % (path, self.path))
    finder = resources.finder_for_path(self.path)
    if finder is None:
        raise DistlibException('Unable to get a finder for %s' % self.path)
    return finder.find(path)

def get_distinfo_file(self, path):
    """
    Returns a path located under the ``.dist-info`` directory. Returns a
    string representing the path.

    :parameter path: a ``'/'``-separated path relative to the
                     ``.dist-info`` directory or an absolute path;
                     If *path* is an absolute path and doesn't start
                     with the ``.dist-info`` directory path,
                     a :class:`DistlibException` is raised
    :type path: str
    :rtype: str
    """
    # Check if it is an absolute path  # XXX use relpath, add tests
    if path.find(os.sep) >= 0:
        # it's an absolute path?
        distinfo_dirname, path = path.split(os.sep)[-2:]
        if distinfo_dirname != self.path.split(os.sep)[-1]:
            raise DistlibException(
                'dist-info file %r does not belong to the %r %s '
                'distribution' % (path, self.name, self.version))

    # The file must be relative
    if path not in DIST_FILES:
        raise DistlibException('invalid path for a dist-info file: '
                               '%r at %r' % (path, self.path))

    return os.path.join(self.path, path)

def list_distinfo_files(self):
    """
    Iterates over the ``RECORD`` entries and returns paths for each line if
    the path is pointing to a file located in the ``.dist-info`` directory
    or one of its subdirectories.

    :returns: iterator of paths
    """
    base = os.path.dirname(self.path)
    for path, checksum, size in self._get_records():
        # XXX add separator or use real relpath algo
        if not os.path.isabs(path):
            path = os.path.join(base, path)
        if path.startswith(self.path):
            yield path

def __eq__(self, other):
    return (isinstance(other, InstalledDistribution)
            and self.path == other.path)

# See http://docs.python.org/reference/datamodel#object.__hash__
__hash__ = object.__hash__


