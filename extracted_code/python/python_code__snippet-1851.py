"""A protocol that various path objects conform.

This exists because importlib.metadata uses both ``pathlib.Path`` and
``zipfile.Path``, and we need a common base for type hints (Union does not
work well since ``zipfile.Path`` is too new for our linter setup).

This does not mean to be exhaustive, but only contains things that present
in both classes *that we need*.
"""

@property
def name(self) -> str:
    raise NotImplementedError()

@property
def parent(self) -> "BasePath":
    raise NotImplementedError()


