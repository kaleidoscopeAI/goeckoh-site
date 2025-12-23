"""Invalid (e.g. corrupt) wheel."""

def __init__(self, location: str, name: str):
    self.location = location
    self.name = name

def __str__(self) -> str:
    return f"Wheel '{self.name}' located at {self.location} is invalid."


