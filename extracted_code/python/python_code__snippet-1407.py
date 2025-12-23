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


