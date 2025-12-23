"""
Map each requirement to the extras that demanded it.
"""

def markers_pass(self, req, extras=None):
    """
    Evaluate markers for req against each extra that
    demanded it.

    Return False if the req has a marker and fails
    evaluation. Otherwise, return True.
    """
    extra_evals = (
        req.marker.evaluate({'extra': extra})
        for extra in self.get(req, ()) + (extras or (None,))
    )
    return not req.marker or any(extra_evals)


