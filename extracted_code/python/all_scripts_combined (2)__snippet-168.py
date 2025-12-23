"""
A generic, immutable data packet that flows between gears.
It contains a payload and metadata about the information's origin and state.
"""
# The data payload of the information packet.
payload: Any
# A dictionary for any metadata.
metadata: Dict[str, Any] = field(default_factory=dict)
# The timestamp of when the information was created.
timestamp: float = field(default_factory=time.time)
# The name of the gear that produced this information.
source_gear: Optional[str] = None

def new(self, payload: Any, source_gear: str) -> Information:
    """
    Creates a new Information object, inheriting metadata but updating
    the payload, source, and timestamp.
    """
    return Information(
        payload=payload,
        metadata=self.metadata.copy(),
        source_gear=source_gear
    )

