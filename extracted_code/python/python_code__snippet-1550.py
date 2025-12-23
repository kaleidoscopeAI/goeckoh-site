from pip._vendor.rich.console import Console

console = Console()
layout = Layout()

layout.split_column(
    Layout(name="header", size=3),
    Layout(ratio=1, name="main"),
    Layout(size=10, name="footer"),
)

layout["main"].split_row(Layout(name="side"), Layout(name="body", ratio=2))

layout["body"].split_row(Layout(name="content", ratio=2), Layout(name="s2"))

layout["s2"].split_column(
    Layout(name="top"), Layout(name="middle"), Layout(name="bottom")
)

layout["side"].split_column(Layout(layout.tree, name="left1"), Layout(name="left2"))

layout["content"].update("foo")

console.print(layout)


