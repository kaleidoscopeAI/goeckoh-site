    from pip._vendor.rich import print
    from pip._vendor.rich.table import Table

    grid = Table("Markup", "Result", padding=(0, 1))

    for markup in MARKUP:
        grid.add_row(Text(markup), markup)

    print(grid)


