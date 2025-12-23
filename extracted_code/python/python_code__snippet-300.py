    from pip._vendor.rich.tree import Tree


class LayoutRender(NamedTuple):
    """An individual layout render."""

    region: Region
    render: List[List[Segment]]


