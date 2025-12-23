def iterkeys(d, **kw):
    return d.iterkeys(**kw)

def itervalues(d, **kw):
    return d.itervalues(**kw)

def iteritems(d, **kw):
    return d.iteritems(**kw)

def iterlists(d, **kw):
    return d.iterlists(**kw)

viewkeys = operator.methodcaller("viewkeys")

viewvalues = operator.methodcaller("viewvalues")

viewitems = operator.methodcaller("viewitems")

