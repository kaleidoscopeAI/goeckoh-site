def _derive_abi():
    parts = ['cp', VER_SUFFIX]
    if sysconfig.get_config_var('Py_DEBUG'):
        parts.append('d')
    if IMP_PREFIX == 'cp':
        vi = sys.version_info[:2]
        if vi < (3, 8):
            wpm = sysconfig.get_config_var('WITH_PYMALLOC')
            if wpm is None:
                wpm = True
            if wpm:
                parts.append('m')
            if vi < (3, 3):
                us = sysconfig.get_config_var('Py_UNICODE_SIZE')
                if us == 4 or (us is None and sys.maxunicode == 0x10FFFF):
                    parts.append('u')
    return ''.join(parts)

ABI = _derive_abi()
del _derive_abi

