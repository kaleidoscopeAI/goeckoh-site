defaults = {
    'delimiter': str(','),  # The strs are used because we need native
    'quotechar': str('"'),  # str in the csv API (2.x won't take
    'lineterminator': str('\n')  # Unicode)
}

def __enter__(self):
    return self

def __exit__(self, *exc_info):
    self.stream.close()


