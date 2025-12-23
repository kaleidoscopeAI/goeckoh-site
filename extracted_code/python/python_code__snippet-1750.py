def __init__(self, causes):
    super(ResolutionImpossible, self).__init__(causes)
    # causes is a list of RequirementInformation objects
    self.causes = causes


