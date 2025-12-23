from .console import Console

c = Console()

from .box import DOUBLE, ROUNDED
from .padding import Padding

p = Panel(
    "Hello, World!",
    title="rich.Panel",
    style="white on blue",
    box=DOUBLE,
    padding=1,
)

c.print()
c.print(p)


