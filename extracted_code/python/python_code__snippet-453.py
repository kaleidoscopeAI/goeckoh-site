    from pip._vendor.rich.console import Console, Group
    from pip._vendor.rich.highlighter import ReprHighlighter
    from pip._vendor.rich.panel import Panel

    highlighter = ReprHighlighter()
    console = Console()

    panel = Panel(
        Group(
            Align.left(highlighter("align='left'")),
            Align.center(highlighter("align='center'")),
            Align.right(highlighter("align='right'")),
        ),
        width=60,
        style="on dark_blue",
        title="Align",
    )

    console.print(
        Align.center(panel, vertical="middle", style="on red", height=console.height)
    )


