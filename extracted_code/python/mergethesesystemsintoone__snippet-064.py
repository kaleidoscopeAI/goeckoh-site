from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
import uvicorn
import requests
import numpy as np
import networkx as nx
from bs4 import BeautifulSoup
from scipy.io import wavfile
from scipy.signal import get_window, stft

