"""Built metadata contains inconsistent information.

This is raised when the metadata contains values (e.g. name and version)
that do not match the information previously obtained from sdist filename,
user-supplied ``#egg=`` value, or an install requirement name.
"""

def __init__(
    self, ireq: "InstallRequirement", field: str, f_val: str, m_val: str
) -> None:
    self.ireq = ireq
    self.field = field
    self.f_val = f_val
    self.m_val = m_val

def __str__(self) -> str:
    return (
        f"Requested {self.ireq} has inconsistent {self.field}: "
        f"expected {self.f_val!r}, but metadata has {self.m_val!r}"
    )


