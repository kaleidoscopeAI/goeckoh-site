MARKUP = [
    "[red]Hello World[/red]",
    "[magenta]Hello [b]World[/b]",
    "[bold]Bold[italic] bold and italic [/bold]italic[/italic]",
    "Click [link=https://www.willmcgugan.com]here[/link] to visit my Blog",
    ":warning-emoji: [bold red blink] DANGER![/]",
]

from pip._vendor.rich import print
from pip._vendor.rich.table import Table

grid = Table("Markup", "Result", padding=(0, 1))

for markup in MARKUP:
    grid.add_row(Text(markup), markup)

print(grid)


