import numpy as np
from typing import Any, Dict, List
from dataclasses import dataclass, field
from PIL import Image
import cv2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import math

