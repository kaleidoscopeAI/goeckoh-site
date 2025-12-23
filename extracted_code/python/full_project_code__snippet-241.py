class EachItem(railroad.Group):
    """
    Custom railroad item to compose a:
    - Group containing a
      - OneOrMore containing a
        - Choice of the elements in the Each
    with the group label indicating that all must be matched
    """

    all_label = "[ALL]"

    def __init__(self, *items):
        choice_item = railroad.Choice(len(items) - 1, *items)
        one_or_more_item = railroad.OneOrMore(item=choice_item)
        super().__init__(one_or_more_item, label=self.all_label)


class AnnotatedItem(railroad.Group):
    """
    Simple subclass of Group that creates an annotation label
    """

    def __init__(self, label: str, item):
        super().__init__(item=item, label="[{}]".format(label) if label else label)


class EditablePartial(Generic[T]):
    """
    Acts like a functools.partial, but can be edited. In other words, it represents a type that hasn't yet been
    constructed.
    """

    # We need this here because the railroad constructors actually transform the data, so can't be called until the
    # entire tree is assembled

    def __init__(self, func: Callable[..., T], args: list, kwargs: dict):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    @classmethod
    def from_call(cls, func: Callable[..., T], *args, **kwargs) -> "EditablePartial[T]":
        """
        If you call this function in the same way that you would call the constructor, it will store the arguments
        as you expect. For example EditablePartial.from_call(Fraction, 1, 3)() == Fraction(1, 3)
        """
        return EditablePartial(func=func, args=list(args), kwargs=kwargs)

    @property
    def name(self):
        return self.kwargs["name"]

    def __call__(self) -> T:
        """
        Evaluate the partial and return the result
        """
        args = self.args.copy()
        kwargs = self.kwargs.copy()

        # This is a helpful hack to allow you to specify varargs parameters (e.g. *args) as keyword args (e.g.
        # args=['list', 'of', 'things'])
        arg_spec = inspect.getfullargspec(self.func)
        if arg_spec.varargs in self.kwargs:
            args += kwargs.pop(arg_spec.varargs)

        return self.func(*args, **kwargs)


def railroad_to_html(diagrams: List[NamedDiagram], embed=False, **kwargs) -> str:
    """
    Given a list of NamedDiagram, produce a single HTML string that visualises those diagrams
    :params kwargs: kwargs to be passed in to the template
    """
    data = []
    for diagram in diagrams:
        if diagram.diagram is None:
            continue
        io = StringIO()
        try:
            css = kwargs.get('css')
            diagram.diagram.writeStandalone(io.write, css=css)
        except AttributeError:
            diagram.diagram.writeSvg(io.write)
        title = diagram.name
        if diagram.index == 0:
            title += " (root)"
        data.append({"title": title, "text": "", "svg": io.getvalue()})

    return template.render(diagrams=data, embed=embed, **kwargs)


def resolve_partial(partial: "EditablePartial[T]") -> T:
    """
    Recursively resolves a collection of Partials into whatever type they are
    """
    if isinstance(partial, EditablePartial):
        partial.args = resolve_partial(partial.args)
        partial.kwargs = resolve_partial(partial.kwargs)
        return partial()
    elif isinstance(partial, list):
        return [resolve_partial(x) for x in partial]
    elif isinstance(partial, dict):
        return {key: resolve_partial(x) for key, x in partial.items()}
    else:
        return partial


def to_railroad(
    element: pyparsing.ParserElement,
    diagram_kwargs: typing.Optional[dict] = None,
    vertical: int = 3,
    show_results_names: bool = False,
    show_groups: bool = False,
