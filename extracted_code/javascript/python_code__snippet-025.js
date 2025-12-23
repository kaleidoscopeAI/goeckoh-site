if isinstance(s, (bytes, bytearray)):
    try:
        s = s.decode('ascii')
    except UnicodeDecodeError:
        raise IDNAError('should pass a unicode string to the function rather than a byte string.')
if uts46:
    s = uts46_remap(s, std3_rules, transitional)
trailing_dot = False
result = []
if strict:
    labels = s.split('.')
else:
    labels = _unicode_dots_re.split(s)
if not labels or labels == ['']:
    raise IDNAError('Empty domain')
if labels[-1] == '':
    del labels[-1]
    trailing_dot = True
for label in labels:
    s = alabel(label)
    if s:
        result.append(s)
    else:
        raise IDNAError('Empty label')
if trailing_dot:
    result.append(b'')
s = b'.'.join(result)
if not valid_string_length(s, trailing_dot):
    raise IDNAError('Domain too long')
return s


