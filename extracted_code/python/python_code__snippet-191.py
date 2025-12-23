    def _make_nmtuple(name, types, module, defaults=()):
        fields = [n for n, t in types]
        annotations = {n: typing._type_check(t, f"field {n} annotation must be a type")
                       for n, t in types}
        nm_tpl = collections.namedtuple(name, fields,
                                        defaults=defaults, module=module)
        nm_tpl.__annotations__ = nm_tpl.__new__.__annotations__ = annotations
        # The `_field_types` attribute was removed in 3.9;
        # in earlier versions, it is the same as the `__annotations__` attribute
        if sys.version_info < (3, 9):
            nm_tpl._field_types = annotations
        return nm_tpl

    _prohibited_namedtuple_fields = typing._prohibited
    _special_namedtuple_fields = frozenset({'__module__', '__name__', '__annotations__'})

    class _NamedTupleMeta(type):
        def __new__(cls, typename, bases, ns):
            assert _NamedTuple in bases
            for base in bases:
                if base is not _NamedTuple and base is not typing.Generic:
                    raise TypeError(
                        'can only inherit from a NamedTuple type and Generic')
            bases = tuple(tuple if base is _NamedTuple else base for base in bases)
            types = ns.get('__annotations__', {})
            default_names = []
            for field_name in types:
                if field_name in ns:
                    default_names.append(field_name)
                elif default_names:
                    raise TypeError(f"Non-default namedtuple field {field_name} "
                                    f"cannot follow default field"
                                    f"{'s' if len(default_names) > 1 else ''} "
                                    f"{', '.join(default_names)}")
            nm_tpl = _make_nmtuple(
                typename, types.items(),
                defaults=[ns[n] for n in default_names],
                module=ns['__module__']
            )
            nm_tpl.__bases__ = bases
            if typing.Generic in bases:
                if hasattr(typing, '_generic_class_getitem'):  # 3.12+
                    nm_tpl.__class_getitem__ = classmethod(typing._generic_class_getitem)
                else:
                    class_getitem = typing.Generic.__class_getitem__.__func__
                    nm_tpl.__class_getitem__ = classmethod(class_getitem)
            # update from user namespace without overriding special namedtuple attributes
            for key in ns:
                if key in _prohibited_namedtuple_fields:
                    raise AttributeError("Cannot overwrite NamedTuple attribute " + key)
                elif key not in _special_namedtuple_fields and key not in nm_tpl._fields:
                    setattr(nm_tpl, key, ns[key])
            if typing.Generic in bases:
                nm_tpl.__init_subclass__()
            return nm_tpl

    _NamedTuple = type.__new__(_NamedTupleMeta, 'NamedTuple', (), {})

    def _namedtuple_mro_entries(bases):
        assert NamedTuple in bases
        return (_NamedTuple,)

    @_ensure_subclassable(_namedtuple_mro_entries)
    def NamedTuple(__typename, __fields=_marker, **kwargs):
        """Typed version of namedtuple.

        Usage::

            class Employee(NamedTuple):
                name: str
                id: int

        This is equivalent to::

            Employee = collections.namedtuple('Employee', ['name', 'id'])

        The resulting class has an extra __annotations__ attribute, giving a
        dict that maps field names to types.  (The field names are also in
        the _fields attribute, which is part of the namedtuple API.)
        An alternative equivalent functional syntax is also accepted::

            Employee = NamedTuple('Employee', [('name', str), ('id', int)])
        """
        if __fields is _marker:
            if kwargs:
                deprecated_thing = "Creating NamedTuple classes using keyword arguments"
                deprecation_msg = (
                    "{name} is deprecated and will be disallowed in Python {remove}. "
                    "Use the class-based or functional syntax instead."
                )
            else:
                deprecated_thing = "Failing to pass a value for the 'fields' parameter"
                example = f"`{__typename} = NamedTuple({__typename!r}, [])`"
                deprecation_msg = (
                    "{name} is deprecated and will be disallowed in Python {remove}. "
                    "To create a NamedTuple class with 0 fields "
                    "using the functional syntax, "
                    "pass an empty list, e.g. "
                ) + example + "."
        elif __fields is None:
            if kwargs:
                raise TypeError(
                    "Cannot pass `None` as the 'fields' parameter "
                    "and also specify fields using keyword arguments"
                )
            else:
                deprecated_thing = "Passing `None` as the 'fields' parameter"
                example = f"`{__typename} = NamedTuple({__typename!r}, [])`"
                deprecation_msg = (
                    "{name} is deprecated and will be disallowed in Python {remove}. "
                    "To create a NamedTuple class with 0 fields "
                    "using the functional syntax, "
                    "pass an empty list, e.g. "
                ) + example + "."
        elif kwargs:
            raise TypeError("Either list of fields or keywords"
                            " can be provided to NamedTuple, not both")
        if __fields is _marker or __fields is None:
            warnings.warn(
                deprecation_msg.format(name=deprecated_thing, remove="3.15"),
                DeprecationWarning,
                stacklevel=2,
            )
            __fields = kwargs.items()
        nt = _make_nmtuple(__typename, __fields, module=_caller())
        nt.__orig_bases__ = (NamedTuple,)
        return nt

    # On 3.8+, alter the signature so that it matches typing.NamedTuple.
    # The signature of typing.NamedTuple on >=3.8 is invalid syntax in Python 3.7,
    # so just leave the signature as it is on 3.7.
    if sys.version_info >= (3, 8):
        _new_signature = '(typename, fields=None, /, **kwargs)'
        if isinstance(NamedTuple, _types.FunctionType):
            NamedTuple.__text_signature__ = _new_signature
        else:
            NamedTuple.__call__.__text_signature__ = _new_signature


