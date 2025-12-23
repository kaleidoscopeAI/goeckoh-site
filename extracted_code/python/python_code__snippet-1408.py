import functools
import itertools


def _nonblank(str):
    return str and not str.startswith("#")


