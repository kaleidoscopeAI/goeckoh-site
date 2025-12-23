<parameter name="code">import numpy as np
from scipy.stats import entropy
def compute_life_intensity(internal_state, environment, lambda1=1.0, lambda2=1.0, lambda3=1.0, tau=1, dt=1, N=1, dN_dt=0):
