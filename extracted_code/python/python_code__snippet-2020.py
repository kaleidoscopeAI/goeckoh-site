try:
    req = get_requirement(req_string)
except InvalidRequirement:
    raise InstallationError(f"Invalid requirement: '{req_string}'")

domains_not_allowed = [
    PyPI.file_storage_domain,
    TestPyPI.file_storage_domain,
]
if (
    req.url
    and comes_from
    and comes_from.link
    and comes_from.link.netloc in domains_not_allowed
):
    # Explicitly disallow pypi packages that depend on external urls
    raise InstallationError(
        "Packages installed from PyPI cannot depend on packages "
        "which are not also hosted on PyPI.\n"
        f"{comes_from.name} depends on {req} "
    )

return InstallRequirement(
    req,
    comes_from,
    isolated=isolated,
    use_pep517=use_pep517,
    user_supplied=user_supplied,
)


