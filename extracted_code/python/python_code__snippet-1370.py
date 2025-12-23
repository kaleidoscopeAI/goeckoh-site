"""
A base class for locators - things that locate distributions.
"""
source_extensions = ('.tar.gz', '.tar.bz2', '.tar', '.zip', '.tgz', '.tbz')
binary_extensions = ('.egg', '.exe', '.whl')
excluded_extensions = ('.pdf',)

# A list of tags indicating which wheels you want to match. The default
# value of None matches against the tags compatible with the running
# Python. If you want to match other values, set wheel_tags on a locator
# instance to a list of tuples (pyver, abi, arch) which you want to match.
wheel_tags = None

downloadable_extensions = source_extensions + ('.whl',)

def __init__(self, scheme='default'):
    """
    Initialise an instance.
    :param scheme: Because locators look for most recent versions, they
                   need to know the version scheme to use. This specifies
                   the current PEP-recommended scheme - use ``'legacy'``
                   if you need to support existing distributions on PyPI.
    """
    self._cache = {}
    self.scheme = scheme
    # Because of bugs in some of the handlers on some of the platforms,
    # we use our own opener rather than just using urlopen.
    self.opener = build_opener(RedirectHandler())
    # If get_project() is called from locate(), the matcher instance
    # is set from the requirement passed to locate(). See issue #18 for
    # why this can be useful to know.
    self.matcher = None
    self.errors = queue.Queue()

def get_errors(self):
    """
    Return any errors which have occurred.
    """
    result = []
    while not self.errors.empty():  # pragma: no cover
        try:
            e = self.errors.get(False)
            result.append(e)
        except self.errors.Empty:
            continue
        self.errors.task_done()
    return result

def clear_errors(self):
    """
    Clear any errors which may have been logged.
    """
    # Just get the errors and throw them away
    self.get_errors()

def clear_cache(self):
    self._cache.clear()

def _get_scheme(self):
    return self._scheme

def _set_scheme(self, value):
    self._scheme = value

scheme = property(_get_scheme, _set_scheme)

def _get_project(self, name):
    """
    For a given project, get a dictionary mapping available versions to Distribution
    instances.

    This should be implemented in subclasses.

    If called from a locate() request, self.matcher will be set to a
    matcher for the requirement to satisfy, otherwise it will be None.
    """
    raise NotImplementedError('Please implement in the subclass')

def get_distribution_names(self):
    """
    Return all the distribution names known to this locator.
    """
    raise NotImplementedError('Please implement in the subclass')

def get_project(self, name):
    """
    For a given project, get a dictionary mapping available versions to Distribution
    instances.

    This calls _get_project to do all the work, and just implements a caching layer on top.
    """
    if self._cache is None:  # pragma: no cover
        result = self._get_project(name)
    elif name in self._cache:
        result = self._cache[name]
    else:
        self.clear_errors()
        result = self._get_project(name)
        self._cache[name] = result
    return result

def score_url(self, url):
    """
    Give an url a score which can be used to choose preferred URLs
    for a given project release.
    """
    t = urlparse(url)
    basename = posixpath.basename(t.path)
    compatible = True
    is_wheel = basename.endswith('.whl')
    is_downloadable = basename.endswith(self.downloadable_extensions)
    if is_wheel:
        compatible = is_compatible(Wheel(basename), self.wheel_tags)
    return (t.scheme == 'https', 'pypi.org' in t.netloc,
            is_downloadable, is_wheel, compatible, basename)

def prefer_url(self, url1, url2):
    """
    Choose one of two URLs where both are candidates for distribution
    archives for the same version of a distribution (for example,
    .tar.gz vs. zip).

    The current implementation favours https:// URLs over http://, archives
    from PyPI over those from other locations, wheel compatibility (if a
    wheel) and then the archive name.
    """
    result = url2
    if url1:
        s1 = self.score_url(url1)
        s2 = self.score_url(url2)
        if s1 > s2:
            result = url1
        if result != url2:
            logger.debug('Not replacing %r with %r', url1, url2)
        else:
            logger.debug('Replacing %r with %r', url1, url2)
    return result

def split_filename(self, filename, project_name):
    """
    Attempt to split a filename in project name, version and Python version.
    """
    return split_filename(filename, project_name)

def convert_url_to_download_info(self, url, project_name):
    """
    See if a URL is a candidate for a download URL for a project (the URL
    has typically been scraped from an HTML page).

    If it is, a dictionary is returned with keys "name", "version",
    "filename" and "url"; otherwise, None is returned.
    """
    def same_project(name1, name2):
        return normalize_name(name1) == normalize_name(name2)

    result = None
    scheme, netloc, path, params, query, frag = urlparse(url)
    if frag.lower().startswith('egg='):  # pragma: no cover
        logger.debug('%s: version hint in fragment: %r',
                     project_name, frag)
    m = HASHER_HASH.match(frag)
    if m:
        algo, digest = m.groups()
    else:
        algo, digest = None, None
    origpath = path
    if path and path[-1] == '/':  # pragma: no cover
        path = path[:-1]
    if path.endswith('.whl'):
        try:
            wheel = Wheel(path)
            if not is_compatible(wheel, self.wheel_tags):
                logger.debug('Wheel not compatible: %s', path)
            else:
                if project_name is None:
                    include = True
                else:
                    include = same_project(wheel.name, project_name)
                if include:
                    result = {
                        'name': wheel.name,
                        'version': wheel.version,
                        'filename': wheel.filename,
                        'url': urlunparse((scheme, netloc, origpath,
                                           params, query, '')),
                        'python-version': ', '.join(
                            ['.'.join(list(v[2:])) for v in wheel.pyver]),
                    }
        except Exception:  # pragma: no cover
            logger.warning('invalid path for wheel: %s', path)
    elif not path.endswith(self.downloadable_extensions):  # pragma: no cover
        logger.debug('Not downloadable: %s', path)
    else:  # downloadable extension
        path = filename = posixpath.basename(path)
        for ext in self.downloadable_extensions:
            if path.endswith(ext):
                path = path[:-len(ext)]
                t = self.split_filename(path, project_name)
                if not t:  # pragma: no cover
                    logger.debug('No match for project/version: %s', path)
                else:
                    name, version, pyver = t
                    if not project_name or same_project(project_name, name):
                        result = {
                            'name': name,
                            'version': version,
                            'filename': filename,
                            'url': urlunparse((scheme, netloc, origpath,
                                               params, query, '')),
                        }
                        if pyver:  # pragma: no cover
                            result['python-version'] = pyver
                break
    if result and algo:
        result['%s_digest' % algo] = digest
    return result

def _get_digest(self, info):
    """
    Get a digest from a dictionary by looking at a "digests" dictionary
    or keys of the form 'algo_digest'.

    Returns a 2-tuple (algo, digest) if found, else None. Currently
    looks only for SHA256, then MD5.
    """
    result = None
    if 'digests' in info:
        digests = info['digests']
        for algo in ('sha256', 'md5'):
            if algo in digests:
                result = (algo, digests[algo])
                break
    if not result:
        for algo in ('sha256', 'md5'):
            key = '%s_digest' % algo
            if key in info:
                result = (algo, info[key])
                break
    return result

def _update_version_data(self, result, info):
    """
    Update a result dictionary (the final result from _get_project) with a
    dictionary for a specific version, which typically holds information
    gleaned from a filename or URL for an archive for the distribution.
    """
    name = info.pop('name')
    version = info.pop('version')
    if version in result:
        dist = result[version]
        md = dist.metadata
    else:
        dist = make_dist(name, version, scheme=self.scheme)
        md = dist.metadata
    dist.digest = digest = self._get_digest(info)
    url = info['url']
    result['digests'][url] = digest
    if md.source_url != info['url']:
        md.source_url = self.prefer_url(md.source_url, url)
        result['urls'].setdefault(version, set()).add(url)
    dist.locator = self
    result[version] = dist

def locate(self, requirement, prereleases=False):
    """
    Find the most recent distribution which matches the given
    requirement.

    :param requirement: A requirement of the form 'foo (1.0)' or perhaps
                        'foo (>= 1.0, < 2.0, != 1.3)'
    :param prereleases: If ``True``, allow pre-release versions
                        to be located. Otherwise, pre-release versions
                        are not returned.
    :return: A :class:`Distribution` instance, or ``None`` if no such
             distribution could be located.
    """
    result = None
    r = parse_requirement(requirement)
    if r is None:  # pragma: no cover
        raise DistlibException('Not a valid requirement: %r' % requirement)
    scheme = get_scheme(self.scheme)
    self.matcher = matcher = scheme.matcher(r.requirement)
    logger.debug('matcher: %s (%s)', matcher, type(matcher).__name__)
    versions = self.get_project(r.name)
    if len(versions) > 2:   # urls and digests keys are present
        # sometimes, versions are invalid
        slist = []
        vcls = matcher.version_class
        for k in versions:
            if k in ('urls', 'digests'):
                continue
            try:
                if not matcher.match(k):
                    pass  # logger.debug('%s did not match %r', matcher, k)
                else:
                    if prereleases or not vcls(k).is_prerelease:
                        slist.append(k)
            except Exception:  # pragma: no cover
                logger.warning('error matching %s with %r', matcher, k)
                pass  # slist.append(k)
        if len(slist) > 1:
            slist = sorted(slist, key=scheme.key)
        if slist:
            logger.debug('sorted list: %s', slist)
            version = slist[-1]
            result = versions[version]
    if result:
        if r.extras:
            result.extras = r.extras
        result.download_urls = versions.get('urls', {}).get(version, set())
        d = {}
        sd = versions.get('digests', {})
        for url in result.download_urls:
            if url in sd:  # pragma: no cover
                d[url] = sd[url]
        result.digests = d
    self.matcher = None
    return result


