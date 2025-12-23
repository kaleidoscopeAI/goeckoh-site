"""Split a layout region in to columns."""

name = "column"

def get_tree_icon(self) -> str:
    return "[layout.tree.column]⬍"

def divide(
    self, children: Sequence["Layout"], region: Region
) -> Iterable[Tuple["Layout", Region]]:
    x, y, width, height = region
    render_heights = ratio_resolve(height, children)
    offset = 0
    _Region = Region
    for child, child_height in zip(children, render_heights):
        yield child, _Region(x, y + offset, width, child_height)
        offset += child_height


