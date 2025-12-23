# For compatibility: Python before 3.9 does not support using [] on the
# Sequence class.
#
# >>> from collections.abc import Sequence
# >>> Sequence[str]
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
# TypeError: 'ABCMeta' object is not subscriptable
#
# TODO: Remove this block after dropping Python 3.8 support.
SequenceCandidate = Sequence


