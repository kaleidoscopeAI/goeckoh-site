"""
Parse a marker string and return a dictionary containing a marker expression.

The dictionary will contain keys "op", "lhs" and "rhs" for non-terminals in
the expression grammar, or strings. A string contained in quotes is to be
interpreted as a literal string, and a string not contained in quotes is a
variable (such as os_name).
"""

def marker_var(remaining):
    # either identifier, or literal string
    m = IDENTIFIER.match(remaining)
    if m:
        result = m.groups()[0]
        remaining = remaining[m.end():]
    elif not remaining:
        raise SyntaxError('unexpected end of input')
    else:
        q = remaining[0]
        if q not in '\'"':
            raise SyntaxError('invalid expression: %s' % remaining)
        oq = '\'"'.replace(q, '')
        remaining = remaining[1:]
        parts = [q]
        while remaining:
            # either a string chunk, or oq, or q to terminate
            if remaining[0] == q:
                break
            elif remaining[0] == oq:
                parts.append(oq)
                remaining = remaining[1:]
            else:
                m = STRING_CHUNK.match(remaining)
                if not m:
                    raise SyntaxError('error in string literal: %s' %
                                      remaining)
                parts.append(m.groups()[0])
                remaining = remaining[m.end():]
        else:
            s = ''.join(parts)
            raise SyntaxError('unterminated string: %s' % s)
        parts.append(q)
        result = ''.join(parts)
        remaining = remaining[1:].lstrip()  # skip past closing quote
    return result, remaining

def marker_expr(remaining):
    if remaining and remaining[0] == '(':
        result, remaining = marker(remaining[1:].lstrip())
        if remaining[0] != ')':
            raise SyntaxError('unterminated parenthesis: %s' % remaining)
        remaining = remaining[1:].lstrip()
    else:
        lhs, remaining = marker_var(remaining)
        while remaining:
            m = MARKER_OP.match(remaining)
            if not m:
                break
            op = m.groups()[0]
            remaining = remaining[m.end():]
            rhs, remaining = marker_var(remaining)
            lhs = {'op': op, 'lhs': lhs, 'rhs': rhs}
        result = lhs
    return result, remaining

def marker_and(remaining):
    lhs, remaining = marker_expr(remaining)
    while remaining:
        m = AND.match(remaining)
        if not m:
            break
        remaining = remaining[m.end():]
        rhs, remaining = marker_expr(remaining)
        lhs = {'op': 'and', 'lhs': lhs, 'rhs': rhs}
    return lhs, remaining

def marker(remaining):
    lhs, remaining = marker_and(remaining)
    while remaining:
        m = OR.match(remaining)
        if not m:
            break
        remaining = remaining[m.end():]
        rhs, remaining = marker_and(remaining)
        lhs = {'op': 'or', 'lhs': lhs, 'rhs': rhs}
    return lhs, remaining

return marker(marker_string)


