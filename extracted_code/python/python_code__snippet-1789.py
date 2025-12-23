"""Metadata provider for egg directories

Usage::

    # Development eggs:

    egg_info = "/path/to/PackageName.egg-info"
    base_dir = os.path.dirname(egg_info)
    metadata = PathMetadata(base_dir, egg_info)
    dist_name = os.path.splitext(os.path.basename(egg_info))[0]
    dist = Distribution(basedir, project_name=dist_name, metadata=metadata)

    # Unpacked egg directories:

    egg_path = "/path/to/PackageName-ver-pyver-etc.egg"
    metadata = PathMetadata(egg_path, os.path.join(egg_path,'EGG-INFO'))
    dist = Distribution.from_filename(egg_path, metadata=metadata)
"""

def __init__(self, path, egg_info):
    self.module_path = path
    self.egg_info = egg_info


