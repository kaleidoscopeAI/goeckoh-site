"""
Creates a Unicode string from a CFString object. Used entirely for error
reporting.

Yes, it annoys me quite a lot that this function is this complex.
"""
value_as_void_p = ctypes.cast(value, ctypes.POINTER(ctypes.c_void_p))

string = CoreFoundation.CFStringGetCStringPtr(
    value_as_void_p, CFConst.kCFStringEncodingUTF8
)
if string is None:
    buffer = ctypes.create_string_buffer(1024)
    result = CoreFoundation.CFStringGetCString(
        value_as_void_p, buffer, 1024, CFConst.kCFStringEncodingUTF8
    )
    if not result:
        raise OSError("Error copying C string from CFStringRef")
    string = buffer.value
if string is not None:
    string = string.decode("utf-8")
return string


