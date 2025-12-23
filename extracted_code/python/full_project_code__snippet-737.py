def __init__(self) -> None:
    super().__init__()
    self.coding_sm = CodingStateMachine(GB2312_SM_MODEL)
    self.distribution_analyzer = GB2312DistributionAnalysis()
    self.reset()

@property
def charset_name(self) -> str:
    return "GB2312"

@property
def language(self) -> str:
    return "Chinese"


