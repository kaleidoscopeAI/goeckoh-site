from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
import numpy as np
import networkx as nx
from bs4 import BeautifulSoup
from scipy.signal import stft, get_window
from sympy import sympify, simplify, Eq, solve
from langdetect import detect, detect_langs

