import numpy as np
from scipy.stats import entropy
from sklearn.decomposition import PCA # For directional mapping

def extract_features(self, D: int, K: int = 5, n_components: int = 1) -> tuple:
    """
    (A) Calculate CA Shannon Entropy (Delta H).
    (B) Calculate Directional Chaos Bias Vector (V_chaos) via PCA.
    
    Args:
        D (int): The dimension of the optimization problem (number of features for the agent).
        K (int): Number of top CA states to consider for entropy calculation.
        n_components (int): Number of principal components to keep (must be >= 1 and <= D).
        
    Returns:
        tuple: (ca_entropy, directional_bias_vector)
    """
    grid = self.grid.flatten()
    
    # 1. Shannon Entropy (Delta H)
    # Use value counts to compute probability distribution P(state)
    states, counts = np.unique(grid, return_counts=True)
    probabilities = counts / counts.sum()
    
    # Calculate Shannon Entropy (natural log, base e)
    ca_entropy = entropy(probabilities, base=2) 
    
    # 2. Directional Chaos Bias (V_chaos)
    # We use the grid itself (flattened) as the feature set for PCA.
    # The grid size (N*N) must be >= D (for meaningful projection).
    if grid.size < D:
        # Fallback for small grids: simple random projection scaled by entropy
        V_chaos = np.random.uniform(-1, 1, D)
        V_chaos = V_chaos / np.linalg.norm(V_chaos) if np.linalg.norm(V_chaos) > 0 else V_chaos
        V_chaos *= ca_entropy # Scale magnitude by entropy
        return ca_entropy, V_chaos

    # Reshape grid for PCA: (N*N, 1) -> (1, N*N)
    # Transpose for feature extraction: (N*N, 1) is a single 'sample' of N*N features
    X = grid.reshape(1, -1) 
    
    # Use PCA to project the 1D feature vector down to a D-dimensional direction
    # We use a linear projection/matrix multiplication instead of a true PCA fit
    # on one sample, which is a mathematical degeneracy.
    # Instead, we define a fixed, rigorous projection.
    
    # The most rigorous approach is to take a linear slice/projection:
    if D <= grid.size:
        # Take the first D elements of the flattened grid, normalize, and center
        V_chaos = grid[:D]
        V_chaos = V_chaos - np.mean(V_chaos)
        
        # Normalize and scale the magnitude by the maximum activity (range) 
        # to ensure the bias vector is proportional to the CA's chaotic intensity.
        activity_range = np.max(grid) - np.min(grid)
        V_chaos_norm = np.linalg.norm(V_chaos)
        if V_chaos_norm > 1e-6:
            V_chaos = V_chaos / V_chaos_norm * activity_range
        
        return ca_entropy, V_chaos
    else:
        # Should not happen if D is properly constrained. Fallback to random projection.
        V_chaos = np.random.uniform(-1, 1, D)
        V_chaos = V_chaos / np.linalg.norm(V_chaos) if np.linalg.norm(V_chaos) > 0 else V_chaos
        return ca_entropy, V_chaos

