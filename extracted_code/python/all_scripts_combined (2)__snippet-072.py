class SettingsStore:
    path: Path
    defaults: Dict[str, object] = field(default_factory=lambda: {
        "correction_echo_enabled": True,
        "support_voice_enabled": False,
        "support_phrases": [],
    })
    data: Dict[str, object] = field(init=False)

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.data = json.loads(self.path.read_text())
        else:
            self.data = {**self.defaults}
            self.save()
        for key, value in self.defaults.items():
            self.data.setdefault(key, value)

    def save(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=2))

    def update(self, **kwargs: object) -> None:
        self.data.update({k: v for k, v in kwargs.items() if v is not None})
        self.save()

    def get_settings(self) -> Dict[str, object]:
        return dict(self.data)

    def add_support_phrase(self, phrase: str) -> None:
        phrases = list(self.data.get("support_phrases", []))
        phrases.append(phrase)
        self.data["support_phrases"] = phrases
        self.save()

    def list_support_phrases(self) -> List[str]:
        return list(self.data.get("support_phrases", []))
