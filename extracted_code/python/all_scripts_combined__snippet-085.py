class AGIStatus:
    """A lightweight snapshot of the AGI state for display purposes."""

    freq_GHz: float = 0.0
    temp_C: float = 0.0
    DA: float = 0.0
    Ser: float = 0.0
    NE: float = 0.0
    coherence: float = 0.0
    awareness: float = 0.0
    phi_proxy: float = 0.0


class KQBCAgent:
    """Agent wrapper around the unified AGI system.

    Each instance owns an ``AGISystem`` and maintains the last log item
    returned from its step function.  The agent provides two main
    methods: ``evaluate_correction`` determines whether an utterance
    requires correction, and ``update_state`` feeds the utterance into
    the AGI and stores the resulting log.  Callers can retrieve a
    simple status dict via ``get_status`` for use in the GUI.
    """

    def __init__(self, config: Optional[CompanionConfig] = None, similarity_threshold: float = 0.78) -> None:
        self.config = config or CompanionConfig()
        self.agi = AGISystem()
        self.similarity_threshold = similarity_threshold
        self.last_log: Optional[Dict[str, object]] = None

    def _string_similarity(self, a: str, b: str) -> float:
        """Compute a simple similarity ratio between two strings.

        This uses Python's built-in ``difflib.SequenceMatcher`` which
        returns a value between 0 and 1 where 1 indicates identical
        strings.  It is sufficient for short phrases and avoids any
        heavy ML dependencies.
        """
        return SequenceMatcher(None, a.strip().lower(), b.strip().lower()).ratio()

    def evaluate_correction(self, target_phrase: str, child_utterance: str) -> bool:
        """Return True if the utterance differs substantially from the target.

        The default implementation computes a simple similarity ratio and
        compares it against ``self.similarity_threshold``.  If the ratio
        falls below the threshold the method returns True to indicate a
        correction should be suggested.
        """
        similarity = self._string_similarity(target_phrase, child_utterance)
        return similarity < self.similarity_threshold

    def update_state(self, user_input: str) -> None:
        """Step the AGI system with the given user input.

        The AGI maintains an internal time and hardware state; each call
        polls sensors, updates thought engines, emotional chemistry and
        relational links, selects an action and logs the result.  The
        resulting log is stored on the agent for later retrieval.
        """
        log_item = self.agi.step(user_input=user_input)
        self.last_log = log_item

    def get_status(self) -> AGIStatus:
        """Return the current AGI status as a plain dataclass.

        If the agent has not yet been updated, returns default values.
        """
        if self.last_log is None:
            return AGIStatus()
        hw = self.last_log.get("hw", {})
        emotion = self.last_log.get("emotion", {})
        consciousness = self.last_log.get("consciousness", {})
        return AGIStatus(
            freq_GHz=float(hw.get("freq_GHz", 0.0)),
            temp_C=float(hw.get("temp_C", 0.0)),
            DA=float(emotion.get("DA", 0.0)),
            Ser=float(emotion.get("Ser", 0.0)),
            NE=float(emotion.get("NE", 0.0)),
            coherence=float(consciousness.get("coherence", 0.0)),
            awareness=float(consciousness.get("awareness", 0.0)),
            phi_proxy=float(consciousness.get("phi_proxy", 0.0)),
        )

