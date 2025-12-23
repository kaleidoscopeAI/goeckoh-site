import numpy as np
import time
import uuid
import asyncio
import logging
from collections import deque, defaultdict
import networkx as nx
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Required for 3D plots
import seaborn as sns
import requests
from quart import Quart, request, jsonify
import threading
