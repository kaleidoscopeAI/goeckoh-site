    from typing import TypedDict

    class ResultDict(TypedDict):
        encoding: Optional[str]
        confidence: float
        language: Optional[str]

