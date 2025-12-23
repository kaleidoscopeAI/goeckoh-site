view = memoryview(obj)
if view.itemsize != 1:
    raise ValueError("cannot unpack from multi-byte object")
return view


