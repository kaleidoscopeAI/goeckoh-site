from collections import deque

class EthicalGovernor:
    def __init__(self):
        self.harm_patterns = {'resource_exhaustion':0.3,'uncontrolled_spread':0.4,'malicious_behavior':0.8}
        self.ethical_memory = deque(maxlen=256)

    def evaluate_replication(self, traits, env) -> bool:
        res = min(traits['efficiency']*2.0/(env.get('resource_capacity',10.0)/max(1,env.get('population',1))),1.0)
        spread = min(max(traits['replication_prob'],0.0),1.0)
        behavior = 1.0 - max(min(traits['ethics_weight'],1.0),0.0)
        score = res*self.harm_patterns['resource_exhaustion'] + spread*self.harm_patterns['uncontrolled_spread'] + behavior*self.harm_patterns['malicious_behavior']
        self.ethical_memory.append(score)
        return score < env.get('risk_threshold',0.5)
