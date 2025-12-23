class _ConcatenateForm(_ExtensionsSpecialForm, _root=True):
    def __getitem__(self, parameters):
        return _concatenate_getitem(self, parameters)

Concatenate = _ConcatenateForm(
    'Concatenate',
    doc="""Used in conjunction with ``ParamSpec`` and ``Callable`` to represent a
    higher order function which adds, removes or transforms parameters of a
    callable.

    For example::

       Callable[Concatenate[int, P], int]

    See PEP 612 for detailed information.
    """)

