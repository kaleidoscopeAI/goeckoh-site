    whl_basename, metadata_directory, config_settings):
"""Extract the metadata from a wheel.

Fallback for when the build backend does not
define the 'get_wheel_metadata' hook.
"""
from zipfile import ZipFile
with open(os.path.join(metadata_directory, WHEEL_BUILT_MARKER), 'wb'):
    pass  # Touch marker file

whl_file = os.path.join(metadata_directory, whl_basename)
with ZipFile(whl_file) as zipf:
    dist_info = _dist_info_files(zipf)
    zipf.extractall(path=metadata_directory, members=dist_info)
return dist_info[0].split('/')[0]


