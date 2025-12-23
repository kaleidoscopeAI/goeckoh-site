Unpack = typing.Unpack

def _is_unpack(obj):
    return get_origin(obj) is Unpack

