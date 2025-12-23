"""
Creates a Unicode string from a CFString object. Used entirely for error
reporting.
Yes, it annoys me quite a lot that this function is this complex.
"""

string = CoreFoundation.CFStringGetCStringPtr(
    cf_string_ref, CFConst.kCFStringEncodingUTF8
)
if string is None:
    buffer = ctypes.create_string_buffer(1024)
    result = CoreFoundation.CFStringGetCString(
        cf_string_ref, buffer, 1024, CFConst.kCFStringEncodingUTF8
    )
    if not result:
        raise OSError("Error copying C string from CFStringRef")
    string = buffer.value
if string is not None:
    string = string.decode("utf-8")
return string  # type: ignore[no-any-return]


