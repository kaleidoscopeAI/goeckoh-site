from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional
import numpy as np
from collections import defaultdict
import networkx as nx
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import time
import enum
from datetime import datetime

class EmotionalState(enum.Enum):
    """Virtual emotional states for enhanced decision-making"""
    NEUTRAL = "neutral"
    ALERT = "alert"        # Low resources or threats
    CURIOUS = "curious"    # New patterns or high confidence
    FOCUSED = "focused"    # Complex learning tasks
    SOCIAL = "social"      # Teaching or collaboration
    CONSERVATIVE = "conservative"  # Resource preservation

