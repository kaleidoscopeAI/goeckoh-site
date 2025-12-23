    import sys

    from pip._vendor.rich.columns import Columns
    from pip._vendor.rich.console import Console

    console = Console(record=True)

    columns = Columns(
        (f":{name}: {name}" for name in sorted(EMOJI.keys()) if "\u200D" not in name),
        column_first=True,
    )

    console.print(columns)
    if len(sys.argv) > 1:
        console.save_html(sys.argv[1])


