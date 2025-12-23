from time import sleep

from .columns import Columns
from .panel import Panel
from .live import Live

all_spinners = Columns(
    [
        Spinner(spinner_name, text=Text(repr(spinner_name), style="green"))
        for spinner_name in sorted(SPINNERS.keys())
    ],
    column_first=True,
    expand=True,
)

with Live(
    Panel(all_spinners, title="Spinners", border_style="blue"),
    refresh_per_second=20,
) as live:
    while True:
        sleep(0.1)


