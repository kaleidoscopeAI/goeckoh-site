"""Representation of possible resolution results of a package.

This holds three attributes:

* `information` is a collection of `RequirementInformation` pairs.
  Each pair is a requirement contributing to this criterion, and the
  candidate that provides the requirement.
* `incompatibilities` is a collection of all known not-to-work candidates
  to exclude from consideration.
* `candidates` is a collection containing all possible candidates deducted
  from the union of contributing requirements and known incompatibilities.
  It should never be empty, except when the criterion is an attribute of a
  raised `RequirementsConflicted` (in which case it is always empty).

.. note::
    This class is intended to be externally immutable. **Do not** mutate
    any of its attribute containers.
"""

def __init__(self, candidates, information, incompatibilities):
    self.candidates = candidates
    self.information = information
    self.incompatibilities = incompatibilities

def __repr__(self):
    requirements = ", ".join(
        "({!r}, via={!r})".format(req, parent)
        for req, parent in self.information
    )
    return "Criterion({})".format(requirements)

def iter_requirement(self):
    return (i.requirement for i in self.information)

def iter_parent(self):
    return (i.parent for i in self.information)


