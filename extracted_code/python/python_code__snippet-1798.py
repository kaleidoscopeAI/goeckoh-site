def _reload_version(self):
    """
    Packages installed by distutils (e.g. numpy or scipy),
    which uses an old safe_version, and so
    their version numbers can get mangled when
    converted to filenames (e.g., 1.11.0.dev0+2329eae to
    1.11.0.dev0_2329eae). These distributions will not be
    parsed properly
    downstream by Distribution and safe_version, so
    take an extra step and try to get the version number from
    the metadata file itself instead of the filename.
    """
    md_version = self._get_version()
    if md_version:
        self._version = md_version
    return self


