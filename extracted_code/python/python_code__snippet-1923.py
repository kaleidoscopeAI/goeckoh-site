"""
Responsible for evaluating links for a particular project.
"""

_py_version_re = re.compile(r"-py([123]\.?[0-9]?)$")

# Don't include an allow_yanked default value to make sure each call
# site considers whether yanked releases are allowed. This also causes
# that decision to be made explicit in the calling code, which helps
# people when reading the code.
def __init__(
    self,
    project_name: str,
    canonical_name: str,
    formats: FrozenSet[str],
    target_python: TargetPython,
    allow_yanked: bool,
    ignore_requires_python: Optional[bool] = None,
) -> None:
    """
    :param project_name: The user supplied package name.
    :param canonical_name: The canonical package name.
    :param formats: The formats allowed for this package. Should be a set
        with 'binary' or 'source' or both in it.
    :param target_python: The target Python interpreter to use when
        evaluating link compatibility. This is used, for example, to
        check wheel compatibility, as well as when checking the Python
        version, e.g. the Python version embedded in a link filename
        (or egg fragment) and against an HTML link's optional PEP 503
        "data-requires-python" attribute.
    :param allow_yanked: Whether files marked as yanked (in the sense
        of PEP 592) are permitted to be candidates for install.
    :param ignore_requires_python: Whether to ignore incompatible
        PEP 503 "data-requires-python" values in HTML links. Defaults
        to False.
    """
    if ignore_requires_python is None:
        ignore_requires_python = False

    self._allow_yanked = allow_yanked
    self._canonical_name = canonical_name
    self._ignore_requires_python = ignore_requires_python
    self._formats = formats
    self._target_python = target_python

    self.project_name = project_name

def evaluate_link(self, link: Link) -> Tuple[LinkType, str]:
    """
    Determine whether a link is a candidate for installation.

    :return: A tuple (result, detail), where *result* is an enum
        representing whether the evaluation found a candidate, or the reason
        why one is not found. If a candidate is found, *detail* will be the
        candidate's version string; if one is not found, it contains the
        reason the link fails to qualify.
    """
    version = None
    if link.is_yanked and not self._allow_yanked:
        reason = link.yanked_reason or "<none given>"
        return (LinkType.yanked, f"yanked for reason: {reason}")

    if link.egg_fragment:
        egg_info = link.egg_fragment
        ext = link.ext
    else:
        egg_info, ext = link.splitext()
        if not ext:
            return (LinkType.format_unsupported, "not a file")
        if ext not in SUPPORTED_EXTENSIONS:
            return (
                LinkType.format_unsupported,
                f"unsupported archive format: {ext}",
            )
        if "binary" not in self._formats and ext == WHEEL_EXTENSION:
            reason = f"No binaries permitted for {self.project_name}"
            return (LinkType.format_unsupported, reason)
        if "macosx10" in link.path and ext == ".zip":
            return (LinkType.format_unsupported, "macosx10 one")
        if ext == WHEEL_EXTENSION:
            try:
                wheel = Wheel(link.filename)
            except InvalidWheelFilename:
                return (
                    LinkType.format_invalid,
                    "invalid wheel filename",
                )
            if canonicalize_name(wheel.name) != self._canonical_name:
                reason = f"wrong project name (not {self.project_name})"
                return (LinkType.different_project, reason)

            supported_tags = self._target_python.get_unsorted_tags()
            if not wheel.supported(supported_tags):
                # Include the wheel's tags in the reason string to
                # simplify troubleshooting compatibility issues.
                file_tags = ", ".join(wheel.get_formatted_file_tags())
                reason = (
                    f"none of the wheel's tags ({file_tags}) are compatible "
                    f"(run pip debug --verbose to show compatible tags)"
                )
                return (LinkType.platform_mismatch, reason)

            version = wheel.version

    # This should be up by the self.ok_binary check, but see issue 2700.
    if "source" not in self._formats and ext != WHEEL_EXTENSION:
        reason = f"No sources permitted for {self.project_name}"
        return (LinkType.format_unsupported, reason)

    if not version:
        version = _extract_version_from_fragment(
            egg_info,
            self._canonical_name,
        )
    if not version:
        reason = f"Missing project version for {self.project_name}"
        return (LinkType.format_invalid, reason)

    match = self._py_version_re.search(version)
    if match:
        version = version[: match.start()]
        py_version = match.group(1)
        if py_version != self._target_python.py_version:
            return (
                LinkType.platform_mismatch,
                "Python version is incorrect",
            )

    supports_python = _check_link_requires_python(
        link,
        version_info=self._target_python.py_version_info,
        ignore_requires_python=self._ignore_requires_python,
    )
    if not supports_python:
        reason = f"{version} Requires-Python {link.requires_python}"
        return (LinkType.requires_python_mismatch, reason)

    logger.debug("Found link %s, version: %s", link, version)

    return (LinkType.candidate, version)


