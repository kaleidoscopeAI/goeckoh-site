"""A tag in console markup."""

name: str
"""The tag name. e.g. 'bold'."""
parameters: Optional[str]
"""Any additional parameters after the name."""

def __str__(self) -> str:
    return (
        self.name if self.parameters is None else f"{self.name} {self.parameters}"
    )

@property
def markup(self) -> str:
    """Get the string representation of this tag."""
    return (
        f"[{self.name}]"
        if self.parameters is None
        else f"[{self.name}={self.parameters}]"
    )


