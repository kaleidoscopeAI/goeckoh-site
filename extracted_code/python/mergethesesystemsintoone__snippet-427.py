import numpy as np
import matplotlib.pyplot as plt
import random
import logging
from typing import Dict, List, Any, Tuple, Optional, Set, Callable, Union
from collections import defaultdict, deque
import time
import uuid
from datetime import datetime
import json
import os
import math
import cv2
from PIL import Image
import networkx as nx
from scipy.spatial.distance import cosine
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import nltk
import spacy
from gensim.models import Word2Vec
from typing import Callable
#from authlib.integrations.requests_client import OAuth2Session #Removed to not use any Auth libraries that might conflict
# from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect  # Removed to not use any ASGI interface that needs extra support. 
# from fastapi.middleware.cors import CORSMiddleware  #Removed for single executable deployment
# import uvicorn #Removed for a single function code package without ASGI interface.
