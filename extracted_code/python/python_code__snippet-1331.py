"""Recursively generate a list of distributions from *dists* that are
required by *dist*.

:param dists: a list of distributions
:param dist: a distribution, member of *dists* for which we are interested
             in finding the dependencies.
"""
if dist not in dists:
    raise DistlibException('given distribution %r is not a member '
                           'of the list' % dist.name)
graph = make_graph(dists)

req = set()  # required distributions
todo = graph.adjacency_list[dist]  # list of nodes we should inspect
seen = set(t[0] for t in todo)  # already added to todo

while todo:
    d = todo.pop()[0]
    req.add(d)
    pred_list = graph.adjacency_list[d]
    for pred in pred_list:
        d = pred[0]
        if d not in req and d not in seen:
            seen.add(d)
            todo.append(pred)
return req


