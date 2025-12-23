"""Return (bom_encoding, input), with any BOM removed from the input."""
if input.startswith(b'\xFF\xFE'):
    return _UTF16LE, input[2:]
if input.startswith(b'\xFE\xFF'):
    return _UTF16BE, input[2:]
if input.startswith(b'\xEF\xBB\xBF'):
    return UTF8, input[3:]
return None, input


