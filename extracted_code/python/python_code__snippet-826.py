    from .console import Console

    console = Console()
    console.print("[bold green]hello world![/bold green]")
    console.print("'[bold green]hello world![/bold green]'")

    console.print(" /foo")
    console.print("/foo/")
    console.print("/foo/bar")
    console.print("foo/bar/baz")

    console.print("/foo/bar/baz?foo=bar+egg&egg=baz")
    console.print("/foo/bar/baz/")
    console.print("/foo/bar/baz/egg")
    console.print("/foo/bar/baz/egg.py")
    console.print("/foo/bar/baz/egg.py word")
    console.print(" /foo/bar/baz/egg.py word")
    console.print("foo /foo/bar/baz/egg.py word")
    console.print("foo /foo/bar/ba._++z/egg+.py word")
    console.print("https://example.org?foo=bar#header")

    console.print(1234567.34)
    console.print(1 / 2)
    console.print(-1 / 123123123123)

    console.print(
        "127.0.1.1 bar 192.168.1.4 2001:0db8:85a3:0000:0000:8a2e:0370:7334 foo"
    )
    import json

    console.print_json(json.dumps(obj={"name": "apple", "count": 1}), indent=None)


