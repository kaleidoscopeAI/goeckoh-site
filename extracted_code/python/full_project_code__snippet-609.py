import json
result = {}
if 'lexer' in requested_items:
    info = {}
    for fullname, names, filenames, mimetypes in get_all_lexers():
        info[fullname] = {
            'aliases': names,
            'filenames': filenames,
            'mimetypes': mimetypes
        }
    result['lexers'] = info

if 'formatter' in requested_items:
    info = {}
    for cls in get_all_formatters():
        doc = docstring_headline(cls)
        info[cls.name] = {
            'aliases': cls.aliases,
            'filenames': cls.filenames,
            'doc': doc
        }
    result['formatters'] = info

if 'filter' in requested_items:
    info = {}
    for name in get_all_filters():
        cls = find_filter_class(name)
        info[name] = {
            'doc': docstring_headline(cls)
        }
    result['filters'] = info

if 'style' in requested_items:
    info = {}
    for name in get_all_styles():
        cls = get_style_by_name(name)
        info[name] = {
            'doc': docstring_headline(cls)
        }
    result['styles'] = info

json.dump(result, sys.stdout)

