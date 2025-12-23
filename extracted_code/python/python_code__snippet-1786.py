"""
Memoized zipfile manifests.
"""

manifest_mod = collections.namedtuple('manifest_mod', 'manifest mtime')

def load(self, path):
    """
    Load a manifest at path or return a suitable manifest already loaded.
    """
    path = os.path.normpath(path)
    mtime = os.stat(path).st_mtime

    if path not in self or self[path].mtime != mtime:
        manifest = self.build(path)
        self[path] = self.manifest_mod(manifest, mtime)

    return self[path].manifest


