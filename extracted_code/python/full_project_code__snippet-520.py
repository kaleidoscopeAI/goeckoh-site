def b(s):
    return s.encode("latin-1")

def u(s):
    return s

unichr = chr
import struct

int2byte = struct.Struct(">B").pack
del struct
byte2int = operator.itemgetter(0)
indexbytes = operator.getitem
iterbytes = iter
import io

StringIO = io.StringIO
BytesIO = io.BytesIO
del io
_assertCountEqual = "assertCountEqual"
if sys.version_info[1] <= 1:
    _assertRaisesRegex = "assertRaisesRegexp"
    _assertRegex = "assertRegexpMatches"
    _assertNotRegex = "assertNotRegexpMatches"
else:
    _assertRaisesRegex = "assertRaisesRegex"
    _assertRegex = "assertRegex"
    _assertNotRegex = "assertNotRegex"
