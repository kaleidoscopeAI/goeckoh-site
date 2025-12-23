def __init__(self):
    self.rules = []

def add_rule(self, rule: Dict):
  """Adds a rule to the rule engine. The input is structured with conditions and an actions for processing of related knowledge ."""
  self.rules.append(rule)

def apply(self, concept: Dict) -> List [Dict]:
  results = []
  for rule in self.rules:
    if self._rule_matches(concept, rule['condition']):
      results.append({"rule_id": rule ["id"], "result": rule["action"] (concept) }) # Appends the rule outcome to the results
  return results

def _rule_matches (self, concept: Dict, condition: Dict) -> bool:
  """ Verifies if conditions are met"""
  for key, value in condition.items ():
        if key == "type": #checks type equality
          if not concept.get ("type") == value:
             return False #type check fail
        elif concept.get (key) != value:
             return False #checks data matches requirement
  return True

