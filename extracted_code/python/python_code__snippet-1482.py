"""Streaming unpacker.

Arguments:

:param file_like:
    File-like object having `.read(n)` method.
    If specified, unpacker reads serialized data from it and :meth:`feed()` is not usable.

:param int read_size:
    Used as `file_like.read(read_size)`. (default: `min(16*1024, max_buffer_size)`)

:param bool use_list:
    If true, unpack msgpack array to Python list.
    Otherwise, unpack to Python tuple. (default: True)

:param bool raw:
    If true, unpack msgpack raw to Python bytes.
    Otherwise, unpack to Python str by decoding with UTF-8 encoding (default).

:param int timestamp:
    Control how timestamp type is unpacked:

        0 - Timestamp
        1 - float  (Seconds from the EPOCH)
        2 - int  (Nanoseconds from the EPOCH)
        3 - datetime.datetime  (UTC).  Python 2 is not supported.

:param bool strict_map_key:
    If true (default), only str or bytes are accepted for map (dict) keys.

:param callable object_hook:
    When specified, it should be callable.
    Unpacker calls it with a dict argument after unpacking msgpack map.
    (See also simplejson)

:param callable object_pairs_hook:
    When specified, it should be callable.
    Unpacker calls it with a list of key-value pairs after unpacking msgpack map.
    (See also simplejson)

:param str unicode_errors:
    The error handler for decoding unicode. (default: 'strict')
    This option should be used only when you have msgpack data which
    contains invalid UTF-8 string.

:param int max_buffer_size:
    Limits size of data waiting unpacked.  0 means 2**32-1.
    The default value is 100*1024*1024 (100MiB).
    Raises `BufferFull` exception when it is insufficient.
    You should set this parameter when unpacking data from untrusted source.

:param int max_str_len:
    Deprecated, use *max_buffer_size* instead.
    Limits max length of str. (default: max_buffer_size)

:param int max_bin_len:
    Deprecated, use *max_buffer_size* instead.
    Limits max length of bin. (default: max_buffer_size)

:param int max_array_len:
    Limits max length of array.
    (default: max_buffer_size)

:param int max_map_len:
    Limits max length of map.
    (default: max_buffer_size//2)

:param int max_ext_len:
    Deprecated, use *max_buffer_size* instead.
    Limits max size of ext type.  (default: max_buffer_size)

Example of streaming deserialize from file-like object::

    unpacker = Unpacker(file_like)
    for o in unpacker:
        process(o)

Example of streaming deserialize from socket::

    unpacker = Unpacker()
    while True:
        buf = sock.recv(1024**2)
        if not buf:
            break
        unpacker.feed(buf)
        for o in unpacker:
            process(o)

Raises ``ExtraData`` when *packed* contains extra bytes.
Raises ``OutOfData`` when *packed* is incomplete.
Raises ``FormatError`` when *packed* is not valid msgpack.
Raises ``StackError`` when *packed* contains too nested.
Other exceptions can be raised during unpacking.
"""

def __init__(
    self,
    file_like=None,
    read_size=0,
    use_list=True,
    raw=False,
    timestamp=0,
    strict_map_key=True,
    object_hook=None,
    object_pairs_hook=None,
    list_hook=None,
    unicode_errors=None,
    max_buffer_size=100 * 1024 * 1024,
    ext_hook=ExtType,
    max_str_len=-1,
    max_bin_len=-1,
    max_array_len=-1,
    max_map_len=-1,
    max_ext_len=-1,
):
    if unicode_errors is None:
        unicode_errors = "strict"

    if file_like is None:
        self._feeding = True
    else:
        if not callable(file_like.read):
            raise TypeError("`file_like.read` must be callable")
        self.file_like = file_like
        self._feeding = False

    #: array of bytes fed.
    self._buffer = bytearray()
    #: Which position we currently reads
    self._buff_i = 0

    # When Unpacker is used as an iterable, between the calls to next(),
    # the buffer is not "consumed" completely, for efficiency sake.
    # Instead, it is done sloppily.  To make sure we raise BufferFull at
    # the correct moments, we have to keep track of how sloppy we were.
    # Furthermore, when the buffer is incomplete (that is: in the case
    # we raise an OutOfData) we need to rollback the buffer to the correct
    # state, which _buf_checkpoint records.
    self._buf_checkpoint = 0

    if not max_buffer_size:
        max_buffer_size = 2**31 - 1
    if max_str_len == -1:
        max_str_len = max_buffer_size
    if max_bin_len == -1:
        max_bin_len = max_buffer_size
    if max_array_len == -1:
        max_array_len = max_buffer_size
    if max_map_len == -1:
        max_map_len = max_buffer_size // 2
    if max_ext_len == -1:
        max_ext_len = max_buffer_size

    self._max_buffer_size = max_buffer_size
    if read_size > self._max_buffer_size:
        raise ValueError("read_size must be smaller than max_buffer_size")
    self._read_size = read_size or min(self._max_buffer_size, 16 * 1024)
    self._raw = bool(raw)
    self._strict_map_key = bool(strict_map_key)
    self._unicode_errors = unicode_errors
    self._use_list = use_list
    if not (0 <= timestamp <= 3):
        raise ValueError("timestamp must be 0..3")
    self._timestamp = timestamp
    self._list_hook = list_hook
    self._object_hook = object_hook
    self._object_pairs_hook = object_pairs_hook
    self._ext_hook = ext_hook
    self._max_str_len = max_str_len
    self._max_bin_len = max_bin_len
    self._max_array_len = max_array_len
    self._max_map_len = max_map_len
    self._max_ext_len = max_ext_len
    self._stream_offset = 0

    if list_hook is not None and not callable(list_hook):
        raise TypeError("`list_hook` is not callable")
    if object_hook is not None and not callable(object_hook):
        raise TypeError("`object_hook` is not callable")
    if object_pairs_hook is not None and not callable(object_pairs_hook):
        raise TypeError("`object_pairs_hook` is not callable")
    if object_hook is not None and object_pairs_hook is not None:
        raise TypeError(
            "object_pairs_hook and object_hook are mutually " "exclusive"
        )
    if not callable(ext_hook):
        raise TypeError("`ext_hook` is not callable")

def feed(self, next_bytes):
    assert self._feeding
    view = _get_data_from_buffer(next_bytes)
    if len(self._buffer) - self._buff_i + len(view) > self._max_buffer_size:
        raise BufferFull

    # Strip buffer before checkpoint before reading file.
    if self._buf_checkpoint > 0:
        del self._buffer[: self._buf_checkpoint]
        self._buff_i -= self._buf_checkpoint
        self._buf_checkpoint = 0

    # Use extend here: INPLACE_ADD += doesn't reliably typecast memoryview in jython
    self._buffer.extend(view)

def _consume(self):
    """Gets rid of the used parts of the buffer."""
    self._stream_offset += self._buff_i - self._buf_checkpoint
    self._buf_checkpoint = self._buff_i

def _got_extradata(self):
    return self._buff_i < len(self._buffer)

def _get_extradata(self):
    return self._buffer[self._buff_i :]

def read_bytes(self, n):
    ret = self._read(n, raise_outofdata=False)
    self._consume()
    return ret

def _read(self, n, raise_outofdata=True):
    # (int) -> bytearray
    self._reserve(n, raise_outofdata=raise_outofdata)
    i = self._buff_i
    ret = self._buffer[i : i + n]
    self._buff_i = i + len(ret)
    return ret

def _reserve(self, n, raise_outofdata=True):
    remain_bytes = len(self._buffer) - self._buff_i - n

    # Fast path: buffer has n bytes already
    if remain_bytes >= 0:
        return

    if self._feeding:
        self._buff_i = self._buf_checkpoint
        raise OutOfData

    # Strip buffer before checkpoint before reading file.
    if self._buf_checkpoint > 0:
        del self._buffer[: self._buf_checkpoint]
        self._buff_i -= self._buf_checkpoint
        self._buf_checkpoint = 0

    # Read from file
    remain_bytes = -remain_bytes
    if remain_bytes + len(self._buffer) > self._max_buffer_size:
        raise BufferFull
    while remain_bytes > 0:
        to_read_bytes = max(self._read_size, remain_bytes)
        read_data = self.file_like.read(to_read_bytes)
        if not read_data:
            break
        assert isinstance(read_data, bytes)
        self._buffer += read_data
        remain_bytes -= len(read_data)

    if len(self._buffer) < n + self._buff_i and raise_outofdata:
        self._buff_i = 0  # rollback
        raise OutOfData

def _read_header(self):
    typ = TYPE_IMMEDIATE
    n = 0
    obj = None
    self._reserve(1)
    b = self._buffer[self._buff_i]
    self._buff_i += 1
    if b & 0b10000000 == 0:
        obj = b
    elif b & 0b11100000 == 0b11100000:
        obj = -1 - (b ^ 0xFF)
    elif b & 0b11100000 == 0b10100000:
        n = b & 0b00011111
        typ = TYPE_RAW
        if n > self._max_str_len:
            raise ValueError("%s exceeds max_str_len(%s)" % (n, self._max_str_len))
        obj = self._read(n)
    elif b & 0b11110000 == 0b10010000:
        n = b & 0b00001111
        typ = TYPE_ARRAY
        if n > self._max_array_len:
            raise ValueError(
                "%s exceeds max_array_len(%s)" % (n, self._max_array_len)
            )
    elif b & 0b11110000 == 0b10000000:
        n = b & 0b00001111
        typ = TYPE_MAP
        if n > self._max_map_len:
            raise ValueError("%s exceeds max_map_len(%s)" % (n, self._max_map_len))
    elif b == 0xC0:
        obj = None
    elif b == 0xC2:
        obj = False
    elif b == 0xC3:
        obj = True
    elif 0xC4 <= b <= 0xC6:
        size, fmt, typ = _MSGPACK_HEADERS[b]
        self._reserve(size)
        if len(fmt) > 0:
            n = _unpack_from(fmt, self._buffer, self._buff_i)[0]
        else:
            n = self._buffer[self._buff_i]
        self._buff_i += size
        if n > self._max_bin_len:
            raise ValueError("%s exceeds max_bin_len(%s)" % (n, self._max_bin_len))
        obj = self._read(n)
    elif 0xC7 <= b <= 0xC9:
        size, fmt, typ = _MSGPACK_HEADERS[b]
        self._reserve(size)
        L, n = _unpack_from(fmt, self._buffer, self._buff_i)
        self._buff_i += size
        if L > self._max_ext_len:
            raise ValueError("%s exceeds max_ext_len(%s)" % (L, self._max_ext_len))
        obj = self._read(L)
    elif 0xCA <= b <= 0xD3:
        size, fmt = _MSGPACK_HEADERS[b]
        self._reserve(size)
        if len(fmt) > 0:
            obj = _unpack_from(fmt, self._buffer, self._buff_i)[0]
        else:
            obj = self._buffer[self._buff_i]
        self._buff_i += size
    elif 0xD4 <= b <= 0xD8:
        size, fmt, typ = _MSGPACK_HEADERS[b]
        if self._max_ext_len < size:
            raise ValueError(
                "%s exceeds max_ext_len(%s)" % (size, self._max_ext_len)
            )
        self._reserve(size + 1)
        n, obj = _unpack_from(fmt, self._buffer, self._buff_i)
        self._buff_i += size + 1
    elif 0xD9 <= b <= 0xDB:
        size, fmt, typ = _MSGPACK_HEADERS[b]
        self._reserve(size)
        if len(fmt) > 0:
            (n,) = _unpack_from(fmt, self._buffer, self._buff_i)
        else:
            n = self._buffer[self._buff_i]
        self._buff_i += size
        if n > self._max_str_len:
            raise ValueError("%s exceeds max_str_len(%s)" % (n, self._max_str_len))
        obj = self._read(n)
    elif 0xDC <= b <= 0xDD:
        size, fmt, typ = _MSGPACK_HEADERS[b]
        self._reserve(size)
        (n,) = _unpack_from(fmt, self._buffer, self._buff_i)
        self._buff_i += size
        if n > self._max_array_len:
            raise ValueError(
                "%s exceeds max_array_len(%s)" % (n, self._max_array_len)
            )
    elif 0xDE <= b <= 0xDF:
        size, fmt, typ = _MSGPACK_HEADERS[b]
        self._reserve(size)
        (n,) = _unpack_from(fmt, self._buffer, self._buff_i)
        self._buff_i += size
        if n > self._max_map_len:
            raise ValueError("%s exceeds max_map_len(%s)" % (n, self._max_map_len))
    else:
        raise FormatError("Unknown header: 0x%x" % b)
    return typ, n, obj

def _unpack(self, execute=EX_CONSTRUCT):
    typ, n, obj = self._read_header()

    if execute == EX_READ_ARRAY_HEADER:
        if typ != TYPE_ARRAY:
            raise ValueError("Expected array")
        return n
    if execute == EX_READ_MAP_HEADER:
        if typ != TYPE_MAP:
            raise ValueError("Expected map")
        return n
    # TODO should we eliminate the recursion?
    if typ == TYPE_ARRAY:
        if execute == EX_SKIP:
            for i in xrange(n):
                # TODO check whether we need to call `list_hook`
                self._unpack(EX_SKIP)
            return
        ret = newlist_hint(n)
        for i in xrange(n):
            ret.append(self._unpack(EX_CONSTRUCT))
        if self._list_hook is not None:
            ret = self._list_hook(ret)
        # TODO is the interaction between `list_hook` and `use_list` ok?
        return ret if self._use_list else tuple(ret)
    if typ == TYPE_MAP:
        if execute == EX_SKIP:
            for i in xrange(n):
                # TODO check whether we need to call hooks
                self._unpack(EX_SKIP)
                self._unpack(EX_SKIP)
            return
        if self._object_pairs_hook is not None:
            ret = self._object_pairs_hook(
                (self._unpack(EX_CONSTRUCT), self._unpack(EX_CONSTRUCT))
                for _ in xrange(n)
            )
        else:
            ret = {}
            for _ in xrange(n):
                key = self._unpack(EX_CONSTRUCT)
                if self._strict_map_key and type(key) not in (unicode, bytes):
                    raise ValueError(
                        "%s is not allowed for map key" % str(type(key))
                    )
                if not PY2 and type(key) is str:
                    key = sys.intern(key)
                ret[key] = self._unpack(EX_CONSTRUCT)
            if self._object_hook is not None:
                ret = self._object_hook(ret)
        return ret
    if execute == EX_SKIP:
        return
    if typ == TYPE_RAW:
        if self._raw:
            obj = bytes(obj)
        else:
            obj = obj.decode("utf_8", self._unicode_errors)
        return obj
    if typ == TYPE_BIN:
        return bytes(obj)
    if typ == TYPE_EXT:
        if n == -1:  # timestamp
            ts = Timestamp.from_bytes(bytes(obj))
            if self._timestamp == 1:
                return ts.to_unix()
            elif self._timestamp == 2:
                return ts.to_unix_nano()
            elif self._timestamp == 3:
                return ts.to_datetime()
            else:
                return ts
        else:
            return self._ext_hook(n, bytes(obj))
    assert typ == TYPE_IMMEDIATE
    return obj

def __iter__(self):
    return self

def __next__(self):
    try:
        ret = self._unpack(EX_CONSTRUCT)
        self._consume()
        return ret
    except OutOfData:
        self._consume()
        raise StopIteration
    except RecursionError:
        raise StackError

next = __next__

def skip(self):
    self._unpack(EX_SKIP)
    self._consume()

def unpack(self):
    try:
        ret = self._unpack(EX_CONSTRUCT)
    except RecursionError:
        raise StackError
    self._consume()
    return ret

def read_array_header(self):
    ret = self._unpack(EX_READ_ARRAY_HEADER)
    self._consume()
    return ret

def read_map_header(self):
    ret = self._unpack(EX_READ_MAP_HEADER)
    self._consume()
    return ret

def tell(self):
    return self._stream_offset


