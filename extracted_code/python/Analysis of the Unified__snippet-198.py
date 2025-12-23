EthicalPrinciple: Stores key metrics like confidence (0.5 initial), applications, success_rate, and context_sensitivity (a dictionary of context keys and their weighted impact).

EthicalViolation: Detailed record used for post-mortem learning, tracking severity, impact_assessment, and mitigation_steps.

EthicalMemory: Manages decisions and consequences, dynamically adjusting the learning_rate (from a minimum of 0.01 to a maximum of 0.5) based on the severity of consequences.

