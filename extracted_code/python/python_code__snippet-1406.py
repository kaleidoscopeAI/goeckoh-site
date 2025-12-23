def __init__(self, **kwargs):
    if 'stream' in kwargs:
        stream = kwargs['stream']
        if sys.version_info[0] >= 3:
            # needs to be a text stream
            stream = codecs.getreader('utf-8')(stream)
        self.stream = stream
    else:
        self.stream = _csv_open(kwargs['path'], 'r')
    self.reader = csv.reader(self.stream, **self.defaults)

def __iter__(self):
    return self

def next(self):
    result = next(self.reader)
    if sys.version_info[0] < 3:
        for i, item in enumerate(result):
            if not isinstance(item, text_type):
                result[i] = item.decode('utf-8')
    return result

__next__ = next


