import unicodedata

categories = {'xid_start': [], 'xid_continue': []}

with open(__file__, encoding='utf-8') as fp:
    content = fp.read()

header = content[:content.find('Cc =')]
footer = content[content.find("def combine("):]

for code in range(0x110000):
    c = chr(code)
    cat = unicodedata.category(c)
    if ord(c) == 0xdc00:
        # Hack to avoid combining this combining with the preceding high
        # surrogate, 0xdbff, when doing a repr.
        c = '\\' + c
    elif ord(c) in (0x2d, 0x5b, 0x5c, 0x5d, 0x5e):
        # Escape regex metachars.
        c = '\\' + c
    categories.setdefault(cat, []).append(c)
    # XID_START and XID_CONTINUE are special categories used for matching
    # identifiers in Python 3.
    if c.isidentifier():
        categories['xid_start'].append(c)
    if ('a' + c).isidentifier():
        categories['xid_continue'].append(c)

with open(__file__, 'w', encoding='utf-8') as fp:
    fp.write(header)

    for cat in sorted(categories):
        val = ''.join(_handle_runs(categories[cat]))
        fp.write('%s = %a\n\n' % (cat, val))

    cats = sorted(categories)
    cats.remove('xid_start')
    cats.remove('xid_continue')
    fp.write('cats = %r\n\n' % cats)

    fp.write('# Generated from unidata %s\n\n' % (unicodedata.unidata_version,))

    fp.write(footer)


