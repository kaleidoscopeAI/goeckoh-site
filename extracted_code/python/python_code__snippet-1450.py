class _UnpackAlias(typing._GenericAlias, _root=True):
    __class__ = typing.TypeVar

class _UnpackForm(_ExtensionsSpecialForm, _root=True):
    def __getitem__(self, parameters):
        item = typing._type_check(parameters,
                                  f'{self._name} accepts only a single type.')
        return _UnpackAlias(self, (item,))

Unpack = _UnpackForm('Unpack', doc=_UNPACK_DOC)

def _is_unpack(obj):
    return isinstance(obj, _UnpackAlias)


