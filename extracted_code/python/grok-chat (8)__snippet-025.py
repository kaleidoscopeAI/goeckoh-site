"""
Possesses an inherent self-monitoring and self-optimizing capability,
designed to detect emergent patterns and proactively adjust the system.
"""
def __init__(self, num_bins=20):
    if not isinstance(num_bins, int) or num_bins <= 0:
        raise ValueError("PatternDetectionTool: num_bins must be a positive integer.")
    self.baseline_distribution = None
    self.bin_edges = None
    self.num_bins = num_bins
    print("--- Pattern Detection Tool Initialized: Listening for reality's emergent harmonies. ---")

def establish_baseline(self, data):
    """
    Establishes a baseline probability distribution from 'normal' data for comparison.
    This is the 'expected harmony'.
    Robustness: Validate input data.
    """
    if not isinstance(data, (list, np.ndarray)) or not data:
        print("  [ERROR] Pattern Detection Directive: Baseline data must be a non-empty list or numpy array.")
        self.baseline_distribution = None
        self.bin_edges = None
        return

    try:
        hist, bin_edges = np.histogram(data, bins=self.num_bins, density=True)
        self.baseline_distribution = hist
        self.bin_edges = bin_edges
        print(f"  Pattern Detection Directive: Established baseline distribution from {len(data)} data points.")
    except Exception as e:
        print(f"  [ERROR] Pattern Detection Directive: Error establishing baseline: {e}")
        self.baseline_distribution = None
        self.bin_edges = None


def detect_emergent_pattern(self, new_data, kl_threshold=0.5):
    """
    Detects anomalies/emergent patterns by comparing new data distribution to the baseline
    using KL Divergence (the 'Harmonic Resonance Engine').
    Robustness: Validate new_data and handle divergence calculation errors.
    """
    if not isinstance(new_data, (list, np.ndarray)) or not new_data:
        print("  [ERROR] Pattern Detection Directive: New data must be a non-empty list or numpy array.")
        return False, None, "Error: Invalid new_data."
    if not isinstance(kl_threshold, (int, float)) or kl_threshold < 0:
        print("  [ERROR] Pattern Detection Directive: kl_threshold must be a non-negative number.")
        return False, None, "Error: Invalid kl_threshold."

    if self.baseline_distribution is None or self.bin_edges is None:
        print("  [WARNING] Pattern Detection Directive: Baseline not established. Cannot detect patterns.")
        return False, None, "No baseline established."

    try:
        # Ensure new_data is within the established bin range or handle out-of-range
        # For simplicity, we'll clip values to the baseline range
        clipped_new_data = np.clip(new_data, self.bin_edges[0], self.bin_edges[-1])
        new_hist, _ = np.histogram(clipped_new_data, bins=self.num_bins, range=(self.bin_edges[0], self.bin_edges[-1]), density=True)

        # Add a small epsilon to avoid log(0) if a bin has zero probability
        epsilon = 1e-10
        p_dist = new_hist + epsilon
        q_dist = self.baseline_distribution + epsilon

        kl_divergence = entropy(p_dist, q_dist) # This is our 'harmonic divergence score'

        is_emergent = kl_divergence > kl_threshold
        print(f"  Pattern Detection Directive: KL Divergence = {kl_divergence:.4f} (Threshold: {kl_threshold}). Emergent Pattern Detected: {is_emergent}")

        # Trigger Conceptual Causal Reasoning Engine and Proactive Self-Optimization
        if is_emergent:
            return is_emergent, kl_divergence, self._trigger_causal_reasoning_and_optimization(new_data, p_dist, q_dist, kl_divergence)
        return is_emergent, kl_divergence, "No specific optimization directive."
    except Exception as e:
        print(f"  [ERROR] Pattern Detection Directive: Error during pattern detection: {e}")
        return False, None, f"Error during detection: {e}"

def _trigger_causal_reasoning_and_optimization(self, new_data, p_dist, q_dist, kl_divergence):
    """
    Conceptual 'Causal Reasoning Engine' that translates detected patterns
    into 'directives' for self-optimization for the Conscious Cube.
    Robustness: Ensure calculations are safe.
    """
    print("\n  Pattern Detection Directive (Causal Reasoning & Self-Optimization):")
    try:
        # Identify the most divergent bins (where p_dist differs most from q_dist)
        # Avoid division by zero if q_dist has zeros (handled by epsilon already)
        ratio_diff = np.log(p_dist / q_dist)
        contributions = p_dist * ratio_diff # Contribution of each bin to total divergence
        # Handle case where contributions might be all NaN/Inf if p_dist or q_dist had issues
        if not np.isfinite(contributions).all():
            print("  [WARNING] Contributions are not finite. Cannot determine most impactful bin.")
            most_impactful_bin_idx = 0 # Default to first bin or handle differently
        else:
            most_impactful_bin_idx = np.argmax(np.abs(contributions)) # Find bin with largest absolute impact

        lower_bound = self.bin_edges[most_impactful_bin_idx]
        upper_bound = self.bin_edges[most_impactful_bin_idx + 1]

        reasoning = f"    **Critical Finding**: An emergent pattern detected with high divergence (KL={kl_divergence:.2f})."
        reasoning += f" Most significant change observed in data range [{lower_bound:.2f}, {upper_bound:.2f}]."

        # Simple conceptual reasoning based on bin location / divergence
        optimization_directive = ""
        if contributions[most_impactful_bin_idx] < 0 and lower_bound < np.mean(self.bin_edges):
            reasoning += "\n    **Inferred Cause**: Data values are significantly *depleted* in lower ranges. This might indicate a systemic bottleneck or resource scarcity."
            optimization_directive = "Directive to Conscious Cube: **PRIORITIZE RESOURCE ALLOCATION ANALYSIS** for relevant Super Nodes. Activate deeper monitoring of 'SupplyChain_SN' and 'Finance_SN' linkages."
        elif contributions[most_impactful_bin_idx] > 0 and upper_bound > np.mean(self.bin_edges):
            reasoning += "\n    **Inferred Cause**: Data values are unusually *concentrated* in higher ranges. This could indicate a surge in demand or unexpected positive anomaly."
            optimization_directive = "Directive to Conscious Cube: **INITIATE SCALABILITY PROTOCOLS** for relevant Super Nodes. Focus synthesis on 'Retail_SN' and 'Logistics_SN' for new market opportunities."
        else:
            reasoning += "\n    **Inferred Cause**: Complex, multi-faceted shift in data distribution. Requires multi-domain analysis."
            optimization_directive = "Directive to Conscious Cube: **ACTIVATE CROSS-DOMAIN SYNTHESIS** for Pattern 'X'. Re-evaluate core 'Energy_SN' connections."

        print(reasoning)
        print(f"    **PROACTIVE SELF-OPTIMIZATION DIRECTIVE**: {optimization_directive}")
        return optimization_directive
    except Exception as e:
        print(f"  [ERROR] Causal Reasoning/Optimization encountered an error: {e}")
        return f"Error during optimization directive generation: {e}"


