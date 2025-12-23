  @staticmethod
  def plan(prompt:str)->List[str]:
      steps=[]
      s=prompt.strip().lower()
      if any(k in s for k in ["prove","show","why","because"]):
          steps += ["Define terms","List axioms/facts","Split into subgoals","Check counterexamples","Synthesize argument"]
      if any(k in s for k in ["design","build","create","implement"]):

