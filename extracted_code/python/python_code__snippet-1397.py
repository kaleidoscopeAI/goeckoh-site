def __init__(self):
    self._preds = {}
    self._succs = {}
    self._nodes = set()  # nodes with no preds/succs

def add_node(self, node):
    self._nodes.add(node)

def remove_node(self, node, edges=False):
    if node in self._nodes:
        self._nodes.remove(node)
    if edges:
        for p in set(self._preds.get(node, ())):
            self.remove(p, node)
        for s in set(self._succs.get(node, ())):
            self.remove(node, s)
        # Remove empties
        for k, v in list(self._preds.items()):
            if not v:
                del self._preds[k]
        for k, v in list(self._succs.items()):
            if not v:
                del self._succs[k]

def add(self, pred, succ):
    assert pred != succ
    self._preds.setdefault(succ, set()).add(pred)
    self._succs.setdefault(pred, set()).add(succ)

def remove(self, pred, succ):
    assert pred != succ
    try:
        preds = self._preds[succ]
        succs = self._succs[pred]
    except KeyError:  # pragma: no cover
        raise ValueError('%r not a successor of anything' % succ)
    try:
        preds.remove(pred)
        succs.remove(succ)
    except KeyError:  # pragma: no cover
        raise ValueError('%r not a successor of %r' % (succ, pred))

def is_step(self, step):
    return (step in self._preds or step in self._succs
            or step in self._nodes)

def get_steps(self, final):
    if not self.is_step(final):
        raise ValueError('Unknown: %r' % final)
    result = []
    todo = []
    seen = set()
    todo.append(final)
    while todo:
        step = todo.pop(0)
        if step in seen:
            # if a step was already seen,
            # move it to the end (so it will appear earlier
            # when reversed on return) ... but not for the
            # final step, as that would be confusing for
            # users
            if step != final:
                result.remove(step)
                result.append(step)
        else:
            seen.add(step)
            result.append(step)
            preds = self._preds.get(step, ())
            todo.extend(preds)
    return reversed(result)

@property
def strong_connections(self):
    # http://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm
    index_counter = [0]
    stack = []
    lowlinks = {}
    index = {}
    result = []

    graph = self._succs

    def strongconnect(node):
        # set the depth index for this node to the smallest unused index
        index[node] = index_counter[0]
        lowlinks[node] = index_counter[0]
        index_counter[0] += 1
        stack.append(node)

        # Consider successors
        try:
            successors = graph[node]
        except Exception:
            successors = []
        for successor in successors:
            if successor not in lowlinks:
                # Successor has not yet been visited
                strongconnect(successor)
                lowlinks[node] = min(lowlinks[node], lowlinks[successor])
            elif successor in stack:
                # the successor is in the stack and hence in the current
                # strongly connected component (SCC)
                lowlinks[node] = min(lowlinks[node], index[successor])

        # If `node` is a root node, pop the stack and generate an SCC
        if lowlinks[node] == index[node]:
            connected_component = []

            while True:
                successor = stack.pop()
                connected_component.append(successor)
                if successor == node:
                    break
            component = tuple(connected_component)
            # storing the result
            result.append(component)

    for node in graph:
        if node not in lowlinks:
            strongconnect(node)

    return result

@property
def dot(self):
    result = ['digraph G {']
    for succ in self._preds:
        preds = self._preds[succ]
        for pred in preds:
            result.append('  %s -> %s;' % (pred, succ))
    for node in self._nodes:
        result.append('  %s;' % node)
    result.append('}')
    return '\n'.join(result)


