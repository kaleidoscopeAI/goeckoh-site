import numpy as np
import pandas as pd
from typing import List, Any

class InstrumentationLogger:
    """
    (E) Tracks all key evolutionary events and state metrics.
    """
    def __init__(self, run_id: str, T_max: int, D: int):
        self.run_id = run_id
        self.T_max = T_max
        self.D = D
        self.log = []
        self.crossover_count = 0
        self.mutation_count = 0

    def increment_crossover(self, count=1):
        self.crossover_count += count

    def increment_mutation(self, count=1):
        self.mutation_count += count

    def calculate_diversity(self, positions: np.ndarray) -> float:
        """
        Calculates Population Diversity as the mean Euclidean distance from the centroid.
        """
        if positions.shape[0] <= 1:
            return 0.0
        centroid = np.mean(positions, axis=0)
        distances = np.linalg.norm(positions - centroid, axis=1)
        return np.mean(distances)

    def log_iteration(self, 
                      t: int, 
                      swarm: 'Swarm', 
                      ca_entropy: float, 
                      dynamic_params: dict,
                      extra_metrics: dict = None):
        """
        Logs the state of the system at iteration t.
        """
        positions = np.array([a.position for a in swarm.agents])
        
        # 1. Diversity (Population Variance)
        diversity_metric = self.calculate_diversity(positions)
        
        # 2. Performance Metrics
        g_best_val = swarm.g_best.value
        
        # 3. Log Entry
        entry = {
            'run_id': self.run_id,
            'iteration': t,
            'g_best_value': g_best_val,
            'population_diversity': diversity_metric,
            'ca_entropy_H': ca_entropy,
            'crossover_count_cumulative': self.crossover_count,
            'mutation_count_cumulative': self.mutation_count,
            'param_w': dynamic_params.get('w'),
            'param_c1': dynamic_params.get('c1'),
            'param_c2': dynamic_params.get('c2'),
            'ca_usage_ablated': swarm.ca_usage_ablated, # Ensure this flag is passed
        }
        
        if extra_metrics:
            entry.update(extra_metrics)

        self.log.append(entry)

    def write_csv(self, filename: str):
        """
        Outputs the collected time-series data to a CSV file.
        """
        df = pd.DataFrame(self.log)
        df.to_csv(filename, index=False)
        print(f"\n[INSTRUMENTATION] Logged {len(self.log)} iterations to {filename}")

