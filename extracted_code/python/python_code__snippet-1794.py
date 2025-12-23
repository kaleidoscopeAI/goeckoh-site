"""
Rebuild module.__path__ ensuring that all entries are ordered
corresponding to their sys.path order
"""
sys_path = [_normalize_cached(p) for p in sys.path]

def safe_sys_path_index(entry):
    """
    Workaround for #520 and #513.
    """
    try:
        return sys_path.index(entry)
    except ValueError:
        return float('inf')

def position_in_sys_path(path):
    """
    Return the ordinal of the path based on its position in sys.path
    """
    path_parts = path.split(os.sep)
    module_parts = package_name.count('.') + 1
    parts = path_parts[:-module_parts]
    return safe_sys_path_index(_normalize_cached(os.sep.join(parts)))

new_path = sorted(orig_path, key=position_in_sys_path)
new_path = [_normalize_cached(p) for p in new_path]

if isinstance(module.__path__, list):
    module.__path__[:] = new_path
else:
    module.__path__ = new_path


