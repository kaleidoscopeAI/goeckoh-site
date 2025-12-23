from .console import Console
from .table import Table
from .text import Text

console = Console()

table = Table(show_footer=False, show_edge=True)
table.add_column("Color", width=10, overflow="ellipsis")
table.add_column("Number", justify="right", style="yellow")
table.add_column("Name", style="green")
table.add_column("Hex", style="blue")
table.add_column("RGB", style="magenta")

colors = sorted((v, k) for k, v in ANSI_COLOR_NAMES.items())
for color_number, name in colors:
    if "grey" in name:
        continue
    color_cell = Text(" " * 10, style=f"on {name}")
    if color_number < 16:
        table.add_row(color_cell, f"{color_number}", Text(f'"{name}"'))
    else:
        color = EIGHT_BIT_PALETTE[color_number]  # type: ignore[has-type]
        table.add_row(
            color_cell, str(color_number), Text(f'"{name}"'), color.hex, color.rgb
        )

console.print(table)


