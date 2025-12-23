import argparse
import io

from pip._vendor.rich.console import Console
from pip._vendor.rich.table import Table
from pip._vendor.rich.text import Text

parser = argparse.ArgumentParser()
parser.add_argument("--html", action="store_true", help="Export as HTML table")
args = parser.parse_args()
html: bool = args.html
console = Console(record=True, width=70, file=io.StringIO()) if html else Console()

table = Table("Name", "Styling")

for style_name, style in DEFAULT_STYLES.items():
    table.add_row(Text(style_name, style=style), str(style))

console.print(table)
if html:
    print(console.export_html(inline_styles=True))


