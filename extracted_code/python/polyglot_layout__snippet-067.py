  @staticmethod
  def plan(prompt:str)->List[str]:
       # simple heuristic planner (no LLM): split goals, produce steps
       steps=[]
       s=prompt.strip()
       if any(k in s.lower() for k in ["prove","show","why","because"]):
           steps += ["Define terms precisely","List known axioms/facts","Transform goal into subgoals","Check counterexamples","Synthesize
