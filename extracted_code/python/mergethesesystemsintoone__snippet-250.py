class RuleEngine:
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
     
class TheoremProver:
    def _init_(self):
         self.theorems = []

    def add_theorem (self, theorem: Dict):
         self.theorems.append (theorem)

    def prove(self, concept: Dict, rule_results: List[Dict]) -> List [Dict]:
        """Applies a set of defined premises that derive into conclusions or actions. if requirements match then theorem holds true
         """
        proofs = []
        for theorem in self.theorems:
             if self._theorem_applicable(concept, rule_results, theorem ["premises"]):
                  proofs.append({'theorem_id': theorem['id'], 'conclusion': theorem["conclusion"](concept, rule_results) })
        return proofs
          
    def _theorem_applicable (self, concept: Dict, rule_results: List[Dict], premises: List[Dict]) -> bool:
         """Determines the set requirements for this theorem"""
         for premise in premises:
            if premise["type"]
