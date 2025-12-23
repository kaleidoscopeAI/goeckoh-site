import os
import time
import json
import signal
import logging
import threading
import numpy as np
import torch
import torch.nn as nn
import networkx as nx
from scipy.integrate import solve_ivp
from concurrent.futures import ThreadPoolExecutor
import requests
from bs4 import BeautifulSoup
import asyncio
import websockets
import sqlite3
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import uuid
import math
from collections import deque, defaultdict
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import evdev  # For real HID
import ctypes  # For Rust

