def encode(self, input, errors='strict'):
    return codecs.charmap_encode(input, errors, encoding_table)

def decode(self, input, errors='strict'):
    return codecs.charmap_decode(input, errors, decoding_table)


