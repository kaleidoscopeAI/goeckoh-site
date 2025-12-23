class RichCast(Protocol):
    """An object that may be 'cast' to a console renderable."""

    def __rich__(
        self,
    ) -> Union["ConsoleRenderable", "RichCast", str]:  # pragma: no cover
        ...


