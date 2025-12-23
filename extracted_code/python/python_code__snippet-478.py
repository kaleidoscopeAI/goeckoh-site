    from .console import Console

    console = Console()
    import sys

    def bar(a: Any) -> None:  # 这是对亚洲语言支持的测试。面对模棱两可的想法，拒绝猜测的诱惑
        one = 1
        print(one / a)

    def foo(a: Any) -> None:
        _rich_traceback_guard = True
        zed = {
            "characters": {
                "Paul Atreides",
                "Vladimir Harkonnen",
                "Thufir Hawat",
                "Duncan Idaho",
            },
            "atomic_types": (None, False, True),
        }
        bar(a)

    def error() -> None:

        try:
            try:
                foo(0)
            except:
                slfkjsldkfj  # type: ignore[name-defined]
        except:
            console.print_exception(show_locals=True)

    error()


