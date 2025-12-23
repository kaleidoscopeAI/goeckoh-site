"""Recursively generate a list of distributions from *dists* that are
dependent on *dist*.

:param dists: a list of distributions
:param dist: a distribution, member of *dists* for which we are interested
"""
if dist not in dists:
    raise DistlibException('given distribution %r is not a member '
                           'of the list' % dist.name)
graph = make_graph(dists)

dep = [dist]  # dependent distributions
todo = graph.reverse_list[dist]  # list of nodes we should inspect

while todo:
    d = todo.pop()
    dep.append(d)
    for succ in graph.reverse_list[d]:
        if succ not in dep:
            todo.append(succ)

dep.pop(0)  # remove dist from dep, was there to prevent infinite loops
return dep


