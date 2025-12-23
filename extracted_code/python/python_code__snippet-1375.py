"""
This class locates distributions in a directory tree.
"""

def __init__(self, path, **kwargs):
    """
    Initialise an instance.
    :param path: The root of the directory tree to search.
    :param kwargs: Passed to the superclass constructor,
                   except for:
                   * recursive - if True (the default), subdirectories are
                     recursed into. If False, only the top-level directory
                     is searched,
    """
    self.recursive = kwargs.pop('recursive', True)
    super(DirectoryLocator, self).__init__(**kwargs)
    path = os.path.abspath(path)
    if not os.path.isdir(path):  # pragma: no cover
        raise DistlibException('Not a directory: %r' % path)
    self.base_dir = path

def should_include(self, filename, parent):
    """
    Should a filename be considered as a candidate for a distribution
    archive? As well as the filename, the directory which contains it
    is provided, though not used by the current implementation.
    """
    return filename.endswith(self.downloadable_extensions)

def _get_project(self, name):
    result = {'urls': {}, 'digests': {}}
    for root, dirs, files in os.walk(self.base_dir):
        for fn in files:
            if self.should_include(fn, root):
                fn = os.path.join(root, fn)
                url = urlunparse(('file', '',
                                  pathname2url(os.path.abspath(fn)),
                                  '', '', ''))
                info = self.convert_url_to_download_info(url, name)
                if info:
                    self._update_version_data(result, info)
        if not self.recursive:
            break
    return result

def get_distribution_names(self):
    """
    Return all the distribution names known to this locator.
    """
    result = set()
    for root, dirs, files in os.walk(self.base_dir):
        for fn in files:
            if self.should_include(fn, root):
                fn = os.path.join(root, fn)
                url = urlunparse(('file', '',
                                  pathname2url(os.path.abspath(fn)),
                                  '', '', ''))
                info = self.convert_url_to_download_info(url, None)
                if info:
                    result.add(info['name'])
        if not self.recursive:
            break
    return result


