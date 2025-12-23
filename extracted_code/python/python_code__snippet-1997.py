from typing import TypedDict

class TransformedHit(TypedDict):
    name: str
    summary: str
    versions: List[str]


