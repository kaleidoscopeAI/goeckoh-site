def __init__(self) -> None:
    super().__init__()
    self.coding_sm = CodingStateMachine(EUCKR_SM_MODEL)
    self.distribution_analyzer = EUCKRDistributionAnalysis()
    self.reset()

@property
def charset_name(self) -> str:
    return "EUC-KR"

@property
def language(self) -> str:
    return "Korean"


