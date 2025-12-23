    from pip._vendor.rich.panel import Panel

    console = Console()

    sponsor_message = Table.grid(padding=1)
    sponsor_message.add_column(style="green", justify="right")
    sponsor_message.add_column(no_wrap=True)

    sponsor_message.add_row(
        "Textualize",
        "[u blue link=https://github.com/textualize]https://github.com/textualize",
    )
    sponsor_message.add_row(
        "Twitter",
        "[u blue link=https://twitter.com/willmcgugan]https://twitter.com/willmcgugan",
    )

    intro_message = Text.from_markup(
        """\
We hope you enjoy using Rich!

