def encode(self, data: str, errors: str = 'strict') -> Tuple[bytes, int]:
    if errors != 'strict':
        raise IDNAError('Unsupported error handling \"{}\"'.format(errors))

    if not data:
        return b"", 0

    return encode(data), len(data)

def decode(self, data: bytes, errors: str = 'strict') -> Tuple[str, int]:
    if errors != 'strict':
        raise IDNAError('Unsupported error handling \"{}\"'.format(errors))

    if not data:
        return '', 0

    return decode(data), len(data)

