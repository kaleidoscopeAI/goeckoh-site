"""Split a layout region in to rows."""

name = "row"

def get_tree_icon(self) -> str:
    return "[layout.tree.row]⬌"

def divide(
    self, children: Sequence["Layout"], region: Region
) -> Iterable[Tuple["Layout", Region]]:
    x, y, width, height = region
    render_widths = ratio_resolve(width, children)
    offset = 0
    _Region = Region
    for child, child_width in zip(children, render_widths):
        yield child, _Region(x + offset, y, child_width, height)
        offset += child_width


