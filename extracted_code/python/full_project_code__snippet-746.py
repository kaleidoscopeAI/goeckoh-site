def __init__(self, lang_filter: LanguageFilter = LanguageFilter.NONE) -> None:
    super().__init__(lang_filter=lang_filter)
    self.probers = [
        UTF8Prober(),
        SJISProber(),
        EUCJPProber(),
        GB2312Prober(),
        EUCKRProber(),
        CP949Prober(),
        Big5Prober(),
        EUCTWProber(),
        JOHABProber(),
    ]
    self.reset()


