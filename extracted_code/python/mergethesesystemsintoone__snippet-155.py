def add_rule(self, rule: Dict):
    """Adds a rule to the rule engine."""
    self.rules.append(rule)

def apply(self, concept: Dict) -> List[Dict]:
    """Applies rules to a concept and returns results."""
    results = []
    for rule in self.rules:
        if self._rule_matches(concept, rule['condition']):
            results.append({'rule_id': rule['id'], 'result': rule['action'](concept)})
    return results

def _rule_matches(self, concept: Dict, condition: Dict) -> bool:
    """Checks if a concept matches a rule condition."""
    for key, value in condition.items():
        if key == 'pattern_type':
            if not any(pattern.get('type') == value for pattern in concept.get('patterns', [])):
                return False
        elif concept.get(key) != value:
            return False
    return True
