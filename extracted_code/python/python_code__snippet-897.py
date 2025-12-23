    from pip._vendor.rich import print
    from pip._vendor.rich.panel import Panel

    panel = Styled(Panel("hello"), "on blue")
    print(panel)


