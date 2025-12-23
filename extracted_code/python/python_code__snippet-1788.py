"""Metadata handler for standalone PKG-INFO files

Usage::

    metadata = FileMetadata("/path/to/PKG-INFO")

This provider rejects all data and metadata requests except for PKG-INFO,
which is treated as existing, and will be the contents of the file at
the provided location.
"""

def __init__(self, path):
    self.path = path

def _get_metadata_path(self, name):
    return self.path

def has_metadata(self, name):
    return name == 'PKG-INFO' and os.path.isfile(self.path)

def get_metadata(self, name):
    if name != 'PKG-INFO':
        raise KeyError("No metadata except PKG-INFO is available")

    with io.open(self.path, encoding='utf-8', errors="replace") as f:
        metadata = f.read()
    self._warn_on_replacement(metadata)
    return metadata

def _warn_on_replacement(self, metadata):
    replacement_char = '�'
    if replacement_char in metadata:
        tmpl = "{self.path} could not be properly decoded in UTF-8"
        msg = tmpl.format(**locals())
        warnings.warn(msg)

def get_metadata_lines(self, name):
    return yield_lines(self.get_metadata(name))


