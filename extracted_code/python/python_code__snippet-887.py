    from .__main__ import make_test_card
    from .console import Console

    console = Console()
    with console.pager(styles=True):
        console.print(make_test_card())


