def _strip_extras(path: str) -> Tuple[str, Optional[str]]:
    m = re.match(r"^(.+)(\[[^\]]+\])$", path)
    extras = None
    if m:
        path_no_extras = m.group(1)
        extras = m.group(2)
    else:
        path_no_extras = path

    return path_no_extras, extras


def convert_extras(extras: Optional[str]) -> Set[str]:
    if not extras:
        return set()
    return get_requirement("placeholder" + extras.lower()).extras


def _set_requirement_extras(req: Requirement, new_extras: Set[str]) -> Requirement:
    """
    Returns a new requirement based on the given one, with the supplied extras. If the
    given requirement already has extras those are replaced (or dropped if no new extras
    are given).
    """
    match: Optional[re.Match[str]] = re.fullmatch(
        # see https://peps.python.org/pep-0508/#complete-grammar
        r"([\w\t .-]+)(\[[^\]]*\])?(.*)",
        str(req),
        flags=re.ASCII,
    )
    # ireq.req is a valid requirement so the regex should always match
    assert (
        match is not None
    ), f"regex match on requirement {req} failed, this should never happen"
    pre: Optional[str] = match.group(1)
    post: Optional[str] = match.group(3)
    assert (
        pre is not None and post is not None
    ), f"regex group selection for requirement {req} failed, this should never happen"
    extras: str = "[%s]" % ",".join(sorted(new_extras)) if new_extras else ""
    return Requirement(f"{pre}{extras}{post}")


def parse_editable(editable_req: str) -> Tuple[Optional[str], str, Set[str]]:
    """Parses an editable requirement into:
        - a requirement name
        - an URL
        - extras
        - editable options
    Accepted requirements:
        svn+http://blahblah@rev#egg=Foobar[baz]&subdirectory=version_subdir
        .[some_extra]
    """

    url = editable_req

    # If a file path is specified with extras, strip off the extras.
    url_no_extras, extras = _strip_extras(url)

    if os.path.isdir(url_no_extras):
        # Treating it as code that has already been checked out
        url_no_extras = path_to_url(url_no_extras)

    if url_no_extras.lower().startswith("file:"):
        package_name = Link(url_no_extras).egg_fragment
        if extras:
            return (
                package_name,
                url_no_extras,
                get_requirement("placeholder" + extras.lower()).extras,
            )
        else:
            return package_name, url_no_extras, set()

    for version_control in vcs:
        if url.lower().startswith(f"{version_control}:"):
            url = f"{version_control}+{url}"
            break

    link = Link(url)

    if not link.is_vcs:
        backends = ", ".join(vcs.all_schemes)
        raise InstallationError(
            f"{editable_req} is not a valid editable requirement. "
            f"It should either be a path to a local project or a VCS URL "
            f"(beginning with {backends})."
        )

    package_name = link.egg_fragment
    if not package_name:
        raise InstallationError(
            "Could not detect requirement name for '{}', please specify one "
            "with #egg=your_package_name".format(editable_req)
        )
    return package_name, url, set()


def check_first_requirement_in_file(filename: str) -> None:
    """Check if file is parsable as a requirements file.

    This is heavily based on ``pkg_resources.parse_requirements``, but
    simplified to just check the first meaningful line.

    :raises InvalidRequirement: If the first meaningful line cannot be parsed
        as an requirement.
    """
    with open(filename, encoding="utf-8", errors="ignore") as f:
        # Create a steppable iterator, so we can handle \-continuations.
        lines = (
            line
            for line in (line.strip() for line in f)
            if line and not line.startswith("#")  # Skip blank lines/comments.
        )

        for line in lines:
            # Drop comments -- a hash without a space may be in a URL.
            if " #" in line:
                line = line[: line.find(" #")]
            # If there is a line continuation, drop it, and append the next line.
            if line.endswith("\\"):
                line = line[:-2].strip() + next(lines, "")
            Requirement(line)
            return


def deduce_helpful_msg(req: str) -> str:
    """Returns helpful msg in case requirements file does not exist,
    or cannot be parsed.

    :params req: Requirements file path
    """
    if not os.path.exists(req):
        return f" File '{req}' does not exist."
    msg = " The path does exist. "
    # Try to parse and check if it is a requirements file.
    try:
        check_first_requirement_in_file(req)
    except InvalidRequirement:
        logger.debug("Cannot parse '%s' as requirements file", req)
    else:
        msg += (
            f"The argument you provided "
            f"({req}) appears to be a"
            f" requirements file. If that is the"
            f" case, use the '-r' flag to install"
            f" the packages specified within it."
        )
    return msg


class RequirementParts:
    def __init__(
        self,
        requirement: Optional[Requirement],
        link: Optional[Link],
        markers: Optional[Marker],
        extras: Set[str],
    ):
        self.requirement = requirement
        self.link = link
        self.markers = markers
        self.extras = extras


def parse_req_from_editable(editable_req: str) -> RequirementParts:
    name, url, extras_override = parse_editable(editable_req)

    if name is not None:
        try:
            req: Optional[Requirement] = Requirement(name)
        except InvalidRequirement:
            raise InstallationError(f"Invalid requirement: '{name}'")
    else:
        req = None

    link = Link(url)

    return RequirementParts(req, link, None, extras_override)


