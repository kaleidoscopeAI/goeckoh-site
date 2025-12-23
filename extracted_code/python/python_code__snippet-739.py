class Requirement:
    """Parse a requirement.

    Parse a given requirement string into its parts, such as name, specifier,
    URL, and extras. Raises InvalidRequirement on a badly-formed requirement
    string.
    """

    # TODO: Can we test whether something is contained within a requirement?
    #       If so how do we do that? Do we need to test against the _name_ of
    #       the thing as well as the version? What about the markers?
    # TODO: Can we normalize the name and extra name?

    def __init__(self, requirement_string: str) -> None:
        try:
            req = REQUIREMENT.parseString(requirement_string)
        except ParseException as e:
            raise InvalidRequirement(
                f'Parse error at "{ requirement_string[e.loc : e.loc + 8]!r}": {e.msg}'
            )

        self.name: str = req.name
        if req.url:
            parsed_url = urllib.parse.urlparse(req.url)
            if parsed_url.scheme == "file":
                if urllib.parse.urlunparse(parsed_url) != req.url:
                    raise InvalidRequirement("Invalid URL given")
            elif not (parsed_url.scheme and parsed_url.netloc) or (
                not parsed_url.scheme and not parsed_url.netloc
            ):
                raise InvalidRequirement(f"Invalid URL: {req.url}")
            self.url: TOptional[str] = req.url
        else:
            self.url = None
        self.extras: Set[str] = set(req.extras.asList() if req.extras else [])
        self.specifier: SpecifierSet = SpecifierSet(req.specifier)
        self.marker: TOptional[Marker] = req.marker if req.marker else None

    def __str__(self) -> str:
        parts: List[str] = [self.name]

        if self.extras:
            formatted_extras = ",".join(sorted(self.extras))
            parts.append(f"[{formatted_extras}]")

        if self.specifier:
            parts.append(str(self.specifier))

        if self.url:
            parts.append(f"@ {self.url}")
            if self.marker:
                parts.append(" ")

        if self.marker:
            parts.append(f"; {self.marker}")

        return "".join(parts)

    def __repr__(self) -> str:
        return f"<Requirement('{self}')>"


