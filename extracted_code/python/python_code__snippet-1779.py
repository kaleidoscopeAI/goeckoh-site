"""Manage resource extraction and packages"""

extraction_path = None

def __init__(self):
    self.cached_files = {}

def resource_exists(self, package_or_requirement, resource_name):
    """Does the named resource exist?"""
    return get_provider(package_or_requirement).has_resource(resource_name)

def resource_isdir(self, package_or_requirement, resource_name):
    """Is the named resource an existing directory?"""
    return get_provider(package_or_requirement).resource_isdir(resource_name)

def resource_filename(self, package_or_requirement, resource_name):
    """Return a true filesystem path for specified resource"""
    return get_provider(package_or_requirement).get_resource_filename(
        self, resource_name
    )

def resource_stream(self, package_or_requirement, resource_name):
    """Return a readable file-like object for specified resource"""
    return get_provider(package_or_requirement).get_resource_stream(
        self, resource_name
    )

def resource_string(self, package_or_requirement, resource_name):
    """Return specified resource as a string"""
    return get_provider(package_or_requirement).get_resource_string(
        self, resource_name
    )

def resource_listdir(self, package_or_requirement, resource_name):
    """List the contents of the named resource directory"""
    return get_provider(package_or_requirement).resource_listdir(resource_name)

def extraction_error(self):
    """Give an error message for problems extracting file(s)"""

    old_exc = sys.exc_info()[1]
    cache_path = self.extraction_path or get_default_cache()

    tmpl = textwrap.dedent(
        """
        Can't extract file(s) to egg cache

        The following error occurred while trying to extract file(s)
        to the Python egg cache:

          {old_exc}

        The Python egg cache directory is currently set to:

          {cache_path}

        Perhaps your account does not have write access to this directory?
        You can change the cache directory by setting the PYTHON_EGG_CACHE
        environment variable to point to an accessible directory.
        """
    ).lstrip()
    err = ExtractionError(tmpl.format(**locals()))
    err.manager = self
    err.cache_path = cache_path
    err.original_error = old_exc
    raise err

def get_cache_path(self, archive_name, names=()):
    """Return absolute location in cache for `archive_name` and `names`

    The parent directory of the resulting path will be created if it does
    not already exist.  `archive_name` should be the base filename of the
    enclosing egg (which may not be the name of the enclosing zipfile!),
    including its ".egg" extension.  `names`, if provided, should be a
    sequence of path name parts "under" the egg's extraction location.

    This method should only be called by resource providers that need to
    obtain an extraction location, and only for names they intend to
    extract, as it tracks the generated names for possible cleanup later.
    """
    extract_path = self.extraction_path or get_default_cache()
    target_path = os.path.join(extract_path, archive_name + '-tmp', *names)
    try:
        _bypass_ensure_directory(target_path)
    except Exception:
        self.extraction_error()

    self._warn_unsafe_extraction_path(extract_path)

    self.cached_files[target_path] = 1
    return target_path

@staticmethod
def _warn_unsafe_extraction_path(path):
    """
    If the default extraction path is overridden and set to an insecure
    location, such as /tmp, it opens up an opportunity for an attacker to
    replace an extracted file with an unauthorized payload. Warn the user
    if a known insecure location is used.

    See Distribute #375 for more details.
    """
    if os.name == 'nt' and not path.startswith(os.environ['windir']):
        # On Windows, permissions are generally restrictive by default
        #  and temp directories are not writable by other users, so
        #  bypass the warning.
        return
    mode = os.stat(path).st_mode
    if mode & stat.S_IWOTH or mode & stat.S_IWGRP:
        msg = (
            "Extraction path is writable by group/others "
            "and vulnerable to attack when "
            "used with get_resource_filename ({path}). "
            "Consider a more secure "
            "location (set with .set_extraction_path or the "
            "PYTHON_EGG_CACHE environment variable)."
        ).format(**locals())
        warnings.warn(msg, UserWarning)

def postprocess(self, tempname, filename):
    """Perform any platform-specific postprocessing of `tempname`

    This is where Mac header rewrites should be done; other platforms don't
    have anything special they should do.

    Resource providers should call this method ONLY after successfully
    extracting a compressed resource.  They must NOT call it on resources
    that are already in the filesystem.

    `tempname` is the current (temporary) name of the file, and `filename`
    is the name it will be renamed to by the caller after this routine
    returns.
    """

    if os.name == 'posix':
        # Make the resource executable
        mode = ((os.stat(tempname).st_mode) | 0o555) & 0o7777
        os.chmod(tempname, mode)

def set_extraction_path(self, path):
    """Set the base path where resources will be extracted to, if needed.

    If you do not call this routine before any extractions take place, the
    path defaults to the return value of ``get_default_cache()``.  (Which
    is based on the ``PYTHON_EGG_CACHE`` environment variable, with various
    platform-specific fallbacks.  See that routine's documentation for more
    details.)

    Resources are extracted to subdirectories of this path based upon
    information given by the ``IResourceProvider``.  You may set this to a
    temporary directory, but then you must call ``cleanup_resources()`` to
    delete the extracted files when done.  There is no guarantee that
    ``cleanup_resources()`` will be able to remove all extracted files.

    (Note: you may not change the extraction path for a given resource
    manager once resources have been extracted, unless you first call
    ``cleanup_resources()``.)
    """
    if self.cached_files:
        raise ValueError("Can't change extraction path, files already extracted")

    self.extraction_path = path

def cleanup_resources(self, force=False):
    """
    Delete all extracted resource files and directories, returning a list
    of the file and directory names that could not be successfully removed.
    This function does not have any concurrency protection, so it should
    generally only be called when the extraction path is a temporary
    directory exclusive to a single process.  This method is not
    automatically called; you must call it explicitly or register it as an
    ``atexit`` function if you wish to ensure cleanup of a temporary
    directory used for extractions.
    """
    # XXX


