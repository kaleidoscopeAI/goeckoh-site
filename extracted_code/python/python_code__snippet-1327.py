"""Created with the *path* of the ``.egg-info`` directory or file provided
to the constructor. It reads the metadata contained in the file itself, or
if the given path happens to be a directory, the metadata is read from the
file ``PKG-INFO`` under that directory."""

requested = True  # as we have no way of knowing, assume it was
shared_locations = {}

def __init__(self, path, env=None):

    def set_name_and_version(s, n, v):
        s.name = n
        s.key = n.lower()  # for case-insensitive comparisons
        s.version = v

    self.path = path
    self.dist_path = env
    if env and env._cache_enabled and path in env._cache_egg.path:
        metadata = env._cache_egg.path[path].metadata
        set_name_and_version(self, metadata.name, metadata.version)
    else:
        metadata = self._get_metadata(path)

        # Need to be set before caching
        set_name_and_version(self, metadata.name, metadata.version)

        if env and env._cache_enabled:
            env._cache_egg.add(self)
    super(EggInfoDistribution, self).__init__(metadata, path, env)

def _get_metadata(self, path):
    requires = None

    def parse_requires_data(data):
        """Create a list of dependencies from a requires.txt file.

        *data*: the contents of a setuptools-produced requires.txt file.
        """
        reqs = []
        lines = data.splitlines()
        for line in lines:
            line = line.strip()
            # sectioned files have bare newlines (separating sections)
            if not line:  # pragma: no cover
                continue
            if line.startswith('['):  # pragma: no cover
                logger.warning(
                    'Unexpected line: quitting requirement scan: %r', line)
                break
            r = parse_requirement(line)
            if not r:  # pragma: no cover
                logger.warning('Not recognised as a requirement: %r', line)
                continue
            if r.extras:  # pragma: no cover
                logger.warning('extra requirements in requires.txt are '
                               'not supported')
            if not r.constraints:
                reqs.append(r.name)
            else:
                cons = ', '.join('%s%s' % c for c in r.constraints)
                reqs.append('%s (%s)' % (r.name, cons))
        return reqs

    def parse_requires_path(req_path):
        """Create a list of dependencies from a requires.txt file.

        *req_path*: the path to a setuptools-produced requires.txt file.
        """

        reqs = []
        try:
            with codecs.open(req_path, 'r', 'utf-8') as fp:
                reqs = parse_requires_data(fp.read())
        except IOError:
            pass
        return reqs

    tl_path = tl_data = None
    if path.endswith('.egg'):
        if os.path.isdir(path):
            p = os.path.join(path, 'EGG-INFO')
            meta_path = os.path.join(p, 'PKG-INFO')
            metadata = Metadata(path=meta_path, scheme='legacy')
            req_path = os.path.join(p, 'requires.txt')
            tl_path = os.path.join(p, 'top_level.txt')
            requires = parse_requires_path(req_path)
        else:
            # FIXME handle the case where zipfile is not available
            zipf = zipimport.zipimporter(path)
            fileobj = StringIO(
                zipf.get_data('EGG-INFO/PKG-INFO').decode('utf8'))
            metadata = Metadata(fileobj=fileobj, scheme='legacy')
            try:
                data = zipf.get_data('EGG-INFO/requires.txt')
                tl_data = zipf.get_data('EGG-INFO/top_level.txt').decode(
                    'utf-8')
                requires = parse_requires_data(data.decode('utf-8'))
            except IOError:
                requires = None
    elif path.endswith('.egg-info'):
        if os.path.isdir(path):
            req_path = os.path.join(path, 'requires.txt')
            requires = parse_requires_path(req_path)
            path = os.path.join(path, 'PKG-INFO')
            tl_path = os.path.join(path, 'top_level.txt')
        metadata = Metadata(path=path, scheme='legacy')
    else:
        raise DistlibException('path must end with .egg-info or .egg, '
                               'got %r' % path)

    if requires:
        metadata.add_requirements(requires)
    # look for top-level modules in top_level.txt, if present
    if tl_data is None:
        if tl_path is not None and os.path.exists(tl_path):
            with open(tl_path, 'rb') as f:
                tl_data = f.read().decode('utf-8')
    if not tl_data:
        tl_data = []
    else:
        tl_data = tl_data.splitlines()
    self.modules = tl_data
    return metadata

def __repr__(self):
    return '<EggInfoDistribution %r %s at %r>' % (self.name, self.version,
                                                  self.path)

def __str__(self):
    return "%s %s" % (self.name, self.version)

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
    record_path = os.path.join(self.path, 'installed-files.txt')
    if os.path.exists(record_path):
        for path, _, _ in self.list_installed_files():
            if path == record_path:
                continue
            if not os.path.exists(path):
                mismatches.append((path, 'exists', True, False))
    return mismatches

def list_installed_files(self):
    """
    Iterates over the ``installed-files.txt`` entries and returns a tuple
    ``(path, hash, size)`` for each line.

    :returns: a list of (path, hash, size)
    """

    def _md5(path):
        f = open(path, 'rb')
        try:
            content = f.read()
        finally:
            f.close()
        return hashlib.md5(content).hexdigest()

    def _size(path):
        return os.stat(path).st_size

    record_path = os.path.join(self.path, 'installed-files.txt')
    result = []
    if os.path.exists(record_path):
        with codecs.open(record_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                p = os.path.normpath(os.path.join(self.path, line))
                # "./" is present as a marker between installed files
                # and installation metadata files
                if not os.path.exists(p):
                    logger.warning('Non-existent file: %s', p)
                    if p.endswith(('.pyc', '.pyo')):
                        continue
                    # otherwise fall through and fail
                if not os.path.isdir(p):
                    result.append((p, _md5(p), _size(p)))
        result.append((record_path, None, None))
    return result

def list_distinfo_files(self, absolute=False):
    """
    Iterates over the ``installed-files.txt`` entries and returns paths for
    each line if the path is pointing to a file located in the
    ``.egg-info`` directory or one of its subdirectories.

    :parameter absolute: If *absolute* is ``True``, each returned path is
                      transformed into a local absolute path. Otherwise the
                      raw value from ``installed-files.txt`` is returned.
    :type absolute: boolean
    :returns: iterator of paths
    """
    record_path = os.path.join(self.path, 'installed-files.txt')
    if os.path.exists(record_path):
        skip = True
        with codecs.open(record_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line == './':
                    skip = False
                    continue
                if not skip:
                    p = os.path.normpath(os.path.join(self.path, line))
                    if p.startswith(self.path):
                        if absolute:
                            yield p
                        else:
                            yield line

def __eq__(self, other):
    return (isinstance(other, EggInfoDistribution)
            and self.path == other.path)

# See http://docs.python.org/reference/datamodel#object.__hash__
__hash__ = object.__hash__


