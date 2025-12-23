"""
Return a `Lexer` subclass instance that's guessed from the text in
`text`. For that, the :meth:`.analyse_text()` method of every known lexer
class is called with the text as argument, and the lexer which returned the
highest value will be instantiated and returned.

:exc:`pygments.util.ClassNotFound` is raised if no lexer thinks it can
handle the content.
"""

if not isinstance(_text, str):
    inencoding = options.get('inencoding', options.get('encoding'))
    if inencoding:
        _text = _text.decode(inencoding or 'utf8')
    else:
        _text, _ = guess_decode(_text)

# try to get a vim modeline first
ft = get_filetype_from_buffer(_text)

if ft is not None:
    try:
        return get_lexer_by_name(ft, **options)
    except ClassNotFound:
        pass

best_lexer = [0.0, None]
for lexer in _iter_lexerclasses():
    rv = lexer.analyse_text(_text)
    if rv == 1.0:
        return lexer(**options)
    if rv > best_lexer[0]:
        best_lexer[:] = (rv, lexer)
if not best_lexer[0] or best_lexer[1] is None:
    raise ClassNotFound('no lexer matching the text found')
return best_lexer[1](**options)


