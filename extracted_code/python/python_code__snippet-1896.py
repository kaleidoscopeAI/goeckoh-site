def __init__(
    self,
    name: str,
    req: str,
    editable: bool,
    comments: Iterable[str] = (),
) -> None:
    self.name = name
    self.canonical_name = canonicalize_name(name)
    self.req = req
    self.editable = editable
    self.comments = comments

@classmethod
def from_dist(cls, dist: BaseDistribution) -> "FrozenRequirement":
    editable = dist.editable
    if editable:
        req, comments = _get_editable_info(dist)
    else:
        comments = []
        direct_url = dist.direct_url
        if direct_url:
            # if PEP 610 metadata is present, use it
            req = direct_url_as_pep440_direct_reference(direct_url, dist.raw_name)
        else:
            # name==version requirement
            req = _format_as_name_version(dist)

    return cls(dist.raw_name, req, editable, comments=comments)

def __str__(self) -> str:
    req = self.req
    if self.editable:
        req = f"-e {req}"
    return "\n".join(list(self.comments) + [str(req)]) + "\n"


