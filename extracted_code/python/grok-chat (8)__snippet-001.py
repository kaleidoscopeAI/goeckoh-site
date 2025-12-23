import numpy as np
import random
import pandas as pd
from statsmodels.tsa.api import grangercausalitytests  # Corrected import for Granger causality tests
from statsmodels.tsa.ar_model import AutoReg
from scipy.stats import entropy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
import logging  # Added for better error handling and logging

