    def _allow_reckless_class_checks(depth=3):
        """Allow instance and class checks for special stdlib modules.
        The abc and functools modules indiscriminately call isinstance() and
        issubclass() on the whole MRO of a user class, which may contain protocols.
        """
        return _caller(depth) in {'abc', 'functools', None}

    def _no_init(self, *args, **kwargs):
        if type(self)._is_protocol:
            raise TypeError('Protocols cannot be instantiated')

    if sys.version_info >= (3, 8):
        # Inheriting from typing._ProtocolMeta isn't actually desirable,
        # but is necessary to allow typing.Protocol and typing_extensions.Protocol
        # to mix without getting TypeErrors about "metaclass conflict"
        _typing_Protocol = typing.Protocol
        _ProtocolMetaBase = type(_typing_Protocol)
    else:
        _typing_Protocol = _marker
        _ProtocolMetaBase = abc.ABCMeta

    class _ProtocolMeta(_ProtocolMetaBase):
        # This metaclass is somewhat unfortunate,
        # but is necessary for several reasons...
        #
        # NOTE: DO NOT call super() in any methods in this class
        # That would call the methods on typing._ProtocolMeta on Python 3.8-3.11
        # and those are slow
        def __new__(mcls, name, bases, namespace, **kwargs):
            if name == "Protocol" and len(bases) < 2:
                pass
            elif {Protocol, _typing_Protocol} & set(bases):
                for base in bases:
                    if not (
                        base in {object, typing.Generic, Protocol, _typing_Protocol}
                        or base.__name__ in _PROTO_ALLOWLIST.get(base.__module__, [])
                        or is_protocol(base)
                    ):
                        raise TypeError(
                            f"Protocols can only inherit from other protocols, "
                            f"got {base!r}"
                        )
            return abc.ABCMeta.__new__(mcls, name, bases, namespace, **kwargs)

        def __init__(cls, *args, **kwargs):
            abc.ABCMeta.__init__(cls, *args, **kwargs)
            if getattr(cls, "_is_protocol", False):
                cls.__protocol_attrs__ = _get_protocol_attrs(cls)
                # PEP 544 prohibits using issubclass()
                # with protocols that have non-method members.
                cls.__callable_proto_members_only__ = all(
                    callable(getattr(cls, attr, None)) for attr in cls.__protocol_attrs__
                )

        def __subclasscheck__(cls, other):
            if cls is Protocol:
                return type.__subclasscheck__(cls, other)
            if (
                getattr(cls, '_is_protocol', False)
                and not _allow_reckless_class_checks()
            ):
                if not isinstance(other, type):
                    # Same error message as for issubclass(1, int).
                    raise TypeError('issubclass() arg 1 must be a class')
                if (
                    not cls.__callable_proto_members_only__
                    and cls.__dict__.get("__subclasshook__") is _proto_hook
                ):
                    raise TypeError(
                        "Protocols with non-method members don't support issubclass()"
                    )
                if not getattr(cls, '_is_runtime_protocol', False):
                    raise TypeError(
                        "Instance and class checks can only be used with "
                        "@runtime_checkable protocols"
                    )
            return abc.ABCMeta.__subclasscheck__(cls, other)

        def __instancecheck__(cls, instance):
            # We need this method for situations where attributes are
            # assigned in __init__.
            if cls is Protocol:
                return type.__instancecheck__(cls, instance)
            if not getattr(cls, "_is_protocol", False):
                # i.e., it's a concrete subclass of a protocol
                return abc.ABCMeta.__instancecheck__(cls, instance)

            if (
                not getattr(cls, '_is_runtime_protocol', False) and
                not _allow_reckless_class_checks()
            ):
                raise TypeError("Instance and class checks can only be used with"
                                " @runtime_checkable protocols")

            if abc.ABCMeta.__instancecheck__(cls, instance):
                return True

            for attr in cls.__protocol_attrs__:
                try:
                    val = inspect.getattr_static(instance, attr)
                except AttributeError:
                    break
                if val is None and callable(getattr(cls, attr, None)):
                    break
            else:
                return True

            return False

        def __eq__(cls, other):
            # Hack so that typing.Generic.__class_getitem__
            # treats typing_extensions.Protocol
            # as equivalent to typing.Protocol on Python 3.8+
            if abc.ABCMeta.__eq__(cls, other) is True:
                return True
            return (
                cls is Protocol and other is getattr(typing, "Protocol", object())
            )

        # This has to be defined, or the abc-module cache
        # complains about classes with this metaclass being unhashable,
        # if we define only __eq__!
        def __hash__(cls) -> int:
            return type.__hash__(cls)

    @classmethod
    def _proto_hook(cls, other):
        if not cls.__dict__.get('_is_protocol', False):
            return NotImplemented

        for attr in cls.__protocol_attrs__:
            for base in other.__mro__:
                # Check if the members appears in the class dictionary...
                if attr in base.__dict__:
                    if base.__dict__[attr] is None:
                        return NotImplemented
                    break

                # ...or in annotations, if it is a sub-protocol.
                annotations = getattr(base, '__annotations__', {})
                if (
                    isinstance(annotations, collections.abc.Mapping)
                    and attr in annotations
                    and is_protocol(other)
                ):
                    break
            else:
                return NotImplemented
        return True

    if sys.version_info >= (3, 8):
        class Protocol(typing.Generic, metaclass=_ProtocolMeta):
            __doc__ = typing.Protocol.__doc__
            __slots__ = ()
            _is_protocol = True
            _is_runtime_protocol = False

            def __init_subclass__(cls, *args, **kwargs):
                super().__init_subclass__(*args, **kwargs)

                # Determine if this is a protocol or a concrete subclass.
                if not cls.__dict__.get('_is_protocol', False):
                    cls._is_protocol = any(b is Protocol for b in cls.__bases__)

                # Set (or override) the protocol subclass hook.
                if '__subclasshook__' not in cls.__dict__:
                    cls.__subclasshook__ = _proto_hook

                # Prohibit instantiation for protocol classes
                if cls._is_protocol and cls.__init__ is Protocol.__init__:
                    cls.__init__ = _no_init

    else:
        class Protocol(metaclass=_ProtocolMeta):
            # There is quite a lot of overlapping code with typing.Generic.
            # Unfortunately it is hard to avoid this on Python <3.8,
            # as the typing module on Python 3.7 doesn't let us subclass typing.Generic!
            """Base class for protocol classes. Protocol classes are defined as::

                class Proto(Protocol):
                    def meth(self) -> int:
                        ...

            Such classes are primarily used with static type checkers that recognize
            structural subtyping (static duck-typing), for example::

                class C:
                    def meth(self) -> int:
                        return 0

                def func(x: Proto) -> int:
                    return x.meth()

                func(C())  # Passes static type check

            See PEP 544 for details. Protocol classes decorated with
            @typing_extensions.runtime_checkable act
            as simple-minded runtime-checkable protocols that check
            only the presence of given attributes, ignoring their type signatures.

            Protocol classes can be generic, they are defined as::

                class GenProto(Protocol[T]):
                    def meth(self) -> T:
                        ...
            """
            __slots__ = ()
            _is_protocol = True
            _is_runtime_protocol = False

            def __new__(cls, *args, **kwds):
                if cls is Protocol:
                    raise TypeError("Type Protocol cannot be instantiated; "
                                    "it can only be used as a base class")
                return super().__new__(cls)

            @typing._tp_cache
            def __class_getitem__(cls, params):
                if not isinstance(params, tuple):
                    params = (params,)
                if not params and cls is not typing.Tuple:
                    raise TypeError(
                        f"Parameter list to {cls.__qualname__}[...] cannot be empty")
                msg = "Parameters to generic types must be types."
                params = tuple(typing._type_check(p, msg) for p in params)
                if cls is Protocol:
                    # Generic can only be subscripted with unique type variables.
                    if not all(isinstance(p, typing.TypeVar) for p in params):
                        i = 0
                        while isinstance(params[i], typing.TypeVar):
                            i += 1
                        raise TypeError(
                            "Parameters to Protocol[...] must all be type variables."
                            f" Parameter {i + 1} is {params[i]}")
                    if len(set(params)) != len(params):
                        raise TypeError(
                            "Parameters to Protocol[...] must all be unique")
                else:
                    # Subscripting a regular Generic subclass.
                    _check_generic(cls, params, len(cls.__parameters__))
                return typing._GenericAlias(cls, params)

            def __init_subclass__(cls, *args, **kwargs):
                if '__orig_bases__' in cls.__dict__:
                    error = typing.Generic in cls.__orig_bases__
                else:
                    error = typing.Generic in cls.__bases__
                if error:
                    raise TypeError("Cannot inherit from plain Generic")
                _maybe_adjust_parameters(cls)

                # Determine if this is a protocol or a concrete subclass.
                if not cls.__dict__.get('_is_protocol', None):
                    cls._is_protocol = any(b is Protocol for b in cls.__bases__)

                # Set (or override) the protocol subclass hook.
                if '__subclasshook__' not in cls.__dict__:
                    cls.__subclasshook__ = _proto_hook

                # Prohibit instantiation for protocol classes
                if cls._is_protocol and cls.__init__ is Protocol.__init__:
                    cls.__init__ = _no_init


