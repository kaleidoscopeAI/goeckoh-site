import numpy as np
import hashlib
import time
import torch
import torch.nn as nn
import networkx as nx
from scipy.integrate import solve_ivp
from concurrent.futures import ThreadPoolExecutor
import requests
from bs4 import BeautifulSoup
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import uuid
import math
from collections import deque, defaultdict
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
