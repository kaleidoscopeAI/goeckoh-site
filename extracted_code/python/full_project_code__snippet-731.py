def __init__(self) -> None:
    super().__init__()
    self.coding_sm = CodingStateMachine(EUCTW_SM_MODEL)
    self.distribution_analyzer = EUCTWDistributionAnalysis()
    self.reset()

@property
def charset_name(self) -> str:
    return "EUC-TW"

@property
def language(self) -> str:
    return "Taiwan"


