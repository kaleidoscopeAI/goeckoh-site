try:
    from importlib.metadata import entry_points
except ImportError:
    try:
        from importlib_metadata import entry_points
    except ImportError:
        try:
            from pip._vendor.pkg_resources import iter_entry_points
        except (ImportError, OSError):
            return []
        else:
            return iter_entry_points(group_name)
groups = entry_points()
if hasattr(groups, 'select'):
    # New interface in Python 3.10 and newer versions of the
    # importlib_metadata backport.
    return groups.select(group=group_name)
else:
    # Older interface, deprecated in Python 3.10 and recent
    # importlib_metadata, but we need it in Python 3.8 and 3.9.
    return groups.get(group_name, [])


