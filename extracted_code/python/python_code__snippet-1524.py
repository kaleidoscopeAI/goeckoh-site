def decode(self, input, final=False):
    return codecs.charmap_decode(input, self.errors, decoding_table)[0]


