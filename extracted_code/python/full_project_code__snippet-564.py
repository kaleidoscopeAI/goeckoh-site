"""
Response length doesn't match expected Content-Length

Subclass of :class:`http.client.IncompleteRead` to allow int value
for ``partial`` to avoid creating large objects on streamed reads.
"""

def __init__(self, partial, expected):
    super(IncompleteRead, self).__init__(partial, expected)

def __repr__(self):
    return "IncompleteRead(%i bytes read, %i more expected)" % (
        self.partial,
        self.expected,
    )


