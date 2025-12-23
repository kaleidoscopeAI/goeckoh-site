from pip._vendor.rich.text import Text

t = Text()
print(isinstance(Text, RichRenderable))
print(isinstance(t, RichRenderable))

class Foo:
    pass

f = Foo()
print(isinstance(f, RichRenderable))
print(isinstance("", RichRenderable))


