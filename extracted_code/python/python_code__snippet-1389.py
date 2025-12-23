if sys.version_info[0] >= 3:
    # needs to be a text stream
    stream = codecs.getreader('utf-8')(stream)
# Try to load as JSON, falling back on legacy format
data = stream.read()
stream = StringIO(data)
try:
    jdata = json.load(stream)
    result = jdata['extensions']['python.exports']['exports']
    for group, entries in result.items():
        for k, v in entries.items():
            s = '%s = %s' % (k, v)
            entry = get_export_entry(s)
            assert entry is not None
            entries[k] = entry
    return result
except Exception:
    stream.seek(0, 0)

def read_stream(cp, stream):
    if hasattr(cp, 'read_file'):
        cp.read_file(stream)
    else:
        cp.readfp(stream)

cp = configparser.ConfigParser()
try:
    read_stream(cp, stream)
except configparser.MissingSectionHeaderError:
    stream.close()
    data = textwrap.dedent(data)
    stream = StringIO(data)
    read_stream(cp, stream)

result = {}
for key in cp.sections():
    result[key] = entries = {}
    for name, value in cp.items(key):
        s = '%s = %s' % (name, value)
        entry = get_export_entry(s)
        assert entry is not None
        # entry.dist = self
        entries[name] = entry
return result


