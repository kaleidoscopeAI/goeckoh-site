"""
First, it checks whether a provided path is an installable directory. If it
is, returns the path.

If false, check if the path is an archive file (such as a .whl).
The function checks if the path is a file. If false, if the path has
an @, it will treat it as a PEP 440 URL requirement and return the path.
"""
if _looks_like_path(name) and os.path.isdir(path):
    if is_installable_dir(path):
        return path_to_url(path)
    # TODO: The is_installable_dir test here might not be necessary
    #       now that it is done in load_pyproject_toml too.
    raise InstallationError(
        f"Directory {name!r} is not installable. Neither 'setup.py' "
        "nor 'pyproject.toml' found."
    )
if not is_archive_file(path):
    return None
if os.path.isfile(path):
    return path_to_url(path)
urlreq_parts = name.split("@", 1)
if len(urlreq_parts) >= 2 and not _looks_like_path(urlreq_parts[0]):
    # If the path contains '@' and the part before it does not look
    # like a path, try to treat it as a PEP 440 URL req instead.
    return None
logger.warning(
    "Requirement %r looks like a filename, but the file does not exist",
    name,
)
return path_to_url(path)


