    from typing import Protocol

    class File(Protocol):
        src_record_path: "RecordPath"
        dest_path: str
        changed: bool

        def save(self) -> None:
            pass


