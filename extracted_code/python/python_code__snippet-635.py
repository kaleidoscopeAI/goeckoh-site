def _csv_open(fn, mode, **kwargs):
    if sys.version_info[0] < 3:
        mode += 'b'
    else:
        kwargs['newline'] = ''
        # Python 3 determines encoding from locale. Force 'utf-8'
        # file encoding to match other forced utf-8 encoding
        kwargs['encoding'] = 'utf-8'
    return open(fn, mode, **kwargs)


class CSVBase(object):
    defaults = {
        'delimiter': str(','),  # The strs are used because we need native
        'quotechar': str('"'),  # str in the csv API (2.x won't take
        'lineterminator': str('\n')  # Unicode)
    }

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.stream.close()


class CSVReader(CSVBase):

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


class CSVWriter(CSVBase):

    def __init__(self, fn, **kwargs):
        self.stream = _csv_open(fn, 'w')
        self.writer = csv.writer(self.stream, **self.defaults)

    def writerow(self, row):
        if sys.version_info[0] < 3:
            r = []
            for item in row:
                if isinstance(item, text_type):
                    item = item.encode('utf-8')
                r.append(item)
            row = r
        self.writer.writerow(row)


