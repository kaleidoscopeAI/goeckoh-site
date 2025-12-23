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


