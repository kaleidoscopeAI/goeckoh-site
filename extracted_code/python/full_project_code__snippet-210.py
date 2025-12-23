# Generated from unidata 11.0.0

def combine(*args):
    return ''.join(globals()[cat] for cat in args)


def allexcept(*args):
    newcats = cats[:]
    for arg in args:
        newcats.remove(arg)
    return ''.join(globals()[cat] for cat in newcats)


def _handle_runs(char_list):  # pragma: no cover
    buf = []
    for c in char_list:
        if len(c) == 1:
            if buf and buf[-1][1] == chr(ord(c)-1):
                buf[-1] = (buf[-1][0], c)
            else:
                buf.append((c, c))
        else:
            buf.append((c, c))
    for a, b in buf:
        if a == b:
            yield a
        else:
            yield '%s-%s' % (a, b)


