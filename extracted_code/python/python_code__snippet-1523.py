def encode(self, input, final=False):
    return codecs.charmap_encode(input, self.errors, encoding_table)[0]


