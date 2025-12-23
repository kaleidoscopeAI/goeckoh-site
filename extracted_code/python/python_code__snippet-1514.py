def iter_decode_to_string(input, fallback_encoding):
    output, _encoding = iter_decode(input, fallback_encoding)
    return ''.join(output)
assert iter_decode_to_string([], 'latin1') == ''
assert iter_decode_to_string([b''], 'latin1') == ''
assert iter_decode_to_string([b'\xe9'], 'latin1') == 'é'
assert iter_decode_to_string([b'hello'], 'latin1') == 'hello'
assert iter_decode_to_string([b'he', b'llo'], 'latin1') == 'hello'
assert iter_decode_to_string([b'hell', b'o'], 'latin1') == 'hello'
assert iter_decode_to_string([b'\xc3\xa9'], 'latin1') == 'Ã©'
assert iter_decode_to_string([b'\xEF\xBB\xBF\xc3\xa9'], 'latin1') == 'é'
assert iter_decode_to_string([
    b'\xEF\xBB\xBF', b'\xc3', b'\xa9'], 'latin1') == 'é'
assert iter_decode_to_string([
    b'\xEF\xBB\xBF', b'a', b'\xc3'], 'latin1') == 'a\uFFFD'
assert iter_decode_to_string([
    b'', b'\xEF', b'', b'', b'\xBB\xBF\xc3', b'\xa9'], 'latin1') == 'é'
assert iter_decode_to_string([b'\xEF\xBB\xBF'], 'latin1') == ''
assert iter_decode_to_string([b'\xEF\xBB'], 'latin1') == 'ï»'
assert iter_decode_to_string([b'\xFE\xFF\x00\xe9'], 'latin1') == 'é'
assert iter_decode_to_string([b'\xFF\xFE\xe9\x00'], 'latin1') == 'é'
assert iter_decode_to_string([
    b'', b'\xFF', b'', b'', b'\xFE\xe9', b'\x00'], 'latin1') == 'é'
assert iter_decode_to_string([
    b'', b'h\xe9', b'llo'], 'x-user-defined') == 'h\uF7E9llo'


