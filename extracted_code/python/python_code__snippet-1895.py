"""
Compute and return values (req, comments) for use in
FrozenRequirement.from_dist().
"""
editable_project_location = dist.editable_project_location
assert editable_project_location
location = os.path.normcase(os.path.abspath(editable_project_location))

from pip._internal.vcs import RemoteNotFoundError, RemoteNotValidError, vcs

vcs_backend = vcs.get_backend_for_dir(location)

if vcs_backend is None:
    display = _format_as_name_version(dist)
    logger.debug(
        'No VCS found for editable requirement "%s" in: %r',
        display,
        location,
    )
    return _EditableInfo(
        requirement=location,
        comments=[f"# Editable install with no version control ({display})"],
    )

vcs_name = type(vcs_backend).__name__

try:
    req = vcs_backend.get_src_requirement(location, dist.raw_name)
except RemoteNotFoundError:
    display = _format_as_name_version(dist)
    return _EditableInfo(
        requirement=location,
        comments=[f"# Editable {vcs_name} install with no remote ({display})"],
    )
except RemoteNotValidError as ex:
    display = _format_as_name_version(dist)
    return _EditableInfo(
        requirement=location,
        comments=[
            f"# Editable {vcs_name} install ({display}) with either a deleted "
            f"local remote or invalid URI:",
            f"# '{ex.url}'",
        ],
    )
except BadCommand:
    logger.warning(
        "cannot determine version of editable source in %s "
        "(%s command not found in path)",
        location,
        vcs_backend.name,
    )
    return _EditableInfo(requirement=location, comments=[])
except InstallationError as exc:
    logger.warning("Error when trying to get requirement for VCS system %s", exc)
else:
    return _EditableInfo(requirement=req, comments=[])

logger.warning("Could not determine repository location of %s", location)

return _EditableInfo(
    requirement=location,
    comments=["## !! Could not determine repository location"],
)


