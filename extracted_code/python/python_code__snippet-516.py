class CFConst:
    """CoreFoundation constants"""

    kCFStringEncodingUTF8 = CFStringEncoding(0x08000100)

    errSecIncompleteCertRevocationCheck = -67635
    errSecHostNameMismatch = -67602
    errSecCertificateExpired = -67818
    errSecNotTrusted = -67843


def _bytes_to_cf_data_ref(value: bytes) -> CFDataRef:  # type: ignore[valid-type]
    return CoreFoundation.CFDataCreate(  # type: ignore[no-any-return]
        CoreFoundation.kCFAllocatorDefault, value, len(value)
    )


def _bytes_to_cf_string(value: bytes) -> CFString:
    """
    Given a Python binary data, create a CFString.
    The string must be CFReleased by the caller.
    """
    c_str = ctypes.c_char_p(value)
    cf_str = CoreFoundation.CFStringCreateWithCString(
        CoreFoundation.kCFAllocatorDefault,
        c_str,
        CFConst.kCFStringEncodingUTF8,
    )
    return cf_str  # type: ignore[no-any-return]


def _cf_string_ref_to_str(cf_string_ref: CFStringRef) -> str | None:  # type: ignore[valid-type]
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


def _der_certs_to_cf_cert_array(certs: list[bytes]) -> CFMutableArrayRef:  # type: ignore[valid-type]
    """Builds a CFArray of SecCertificateRefs from a list of DER-encoded certificates.
    Responsibility of the caller to call CoreFoundation.CFRelease on the CFArray.
    """
    cf_array = CoreFoundation.CFArrayCreateMutable(
        CoreFoundation.kCFAllocatorDefault,
        0,
        ctypes.byref(CoreFoundation.kCFTypeArrayCallBacks),
    )
    if not cf_array:
        raise MemoryError("Unable to allocate memory!")

    for cert_data in certs:
        cf_data = None
        sec_cert_ref = None
        try:
            cf_data = _bytes_to_cf_data_ref(cert_data)
            sec_cert_ref = Security.SecCertificateCreateWithData(
                CoreFoundation.kCFAllocatorDefault, cf_data
            )
            CoreFoundation.CFArrayAppendValue(cf_array, sec_cert_ref)
        finally:
            if cf_data:
                CoreFoundation.CFRelease(cf_data)
            if sec_cert_ref:
                CoreFoundation.CFRelease(sec_cert_ref)

    return cf_array  # type: ignore[no-any-return]


