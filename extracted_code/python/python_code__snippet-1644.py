@auto
class Foo:
    def __rich_repr__(self) -> Result:
        yield "foo"
        yield "bar", {"shopping": ["eggs", "ham", "pineapple"]}
        yield "buy", "hand sanitizer"

foo = Foo()
from pip._vendor.rich.console import Console

console = Console()

console.rule("Standard repr")
console.print(foo)

console.print(foo, width=60)
console.print(foo, width=30)

console.rule("Angular repr")
Foo.__rich_repr__.angular = True  # type: ignore[attr-defined]

console.print(foo)

console.print(foo, width=60)
console.print(foo, width=30)


