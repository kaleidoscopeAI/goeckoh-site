    from pip._vendor.rich.console import Console

    text = Text(
        """\nLorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.\n"""
    )
    text.highlight_words(["Lorem"], "bold")
    text.highlight_words(["ipsum"], "italic")

    console = Console()

    console.rule("justify='left'")
    console.print(text, style="red")
    console.print()

    console.rule("justify='center'")
    console.print(text, style="green", justify="center")
    console.print()

    console.rule("justify='right'")
    console.print(text, style="blue", justify="right")
    console.print()

    console.rule("justify='full'")
    console.print(text, style="magenta", justify="full")
    console.print()


