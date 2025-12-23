#!/usr/bin/env python3
# Seed-Crystal AGI: Full Integrated System (Production-Ready)
# Integrates ingestion, annealing, sonification, attention, captioning in autonomous loop.
# Math: Bit embeddings → annealed crystals → energetics (H_bits, S_field, L) → sonify → STFT → multi-head attention → captions.
# Autonomous: Periodically searches X/web for data to ingest.

import os, sys, subprocess, venv, json, time, asyncio, math, sqlite3, textwrap, base64, traceback, struct, wave, random
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Iterable
from pathlib import Path

# Bootstrapping: create .venv and re-exec
ROOT = Path.cwd() / "seed_crystal_agi"
VENV = ROOT / ".venv"
REQ = [
  "fastapi==0.115.5", "uvicorn==0.32.0", "requests==2.32.3", "beautifulsoup4==4.12.3",
  "networkx==3.3", "numpy==1.26.4", "pydantic==2.9.2", "starlette==0.41.3", "websockets==12.0"
]
def ensure_venv_and_reexec():
  ROOT.mkdir(parents=True, exist_ok=True)
  if os.environ.get("SC_BOOTED") == "1":
     return
  if not VENV.exists():
     print("Creating venv at", VENV)
     venv.create(VENV, with_pip=True)
  pip = VENV / ("Scripts/pip.exe" if os.name == "nt" else "bin/pip")
  py = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
  print("Upgrading pip and installing deps")
  subprocess.check_call([str(pip), "install", "--upgrade", "pip"])
  subprocess.check_call([str(pip), "install"] + REQ)
  env = os.environ.copy(); env["SC_BOOTED"] = "1"
  print("Relaunching inside venv")
  os.execvpe(str(py), [str(py), __file__], env)

if __file__ == "<stdin>":
   script_path = ROOT / "seed_crystal_agi.py"
   content = sys.stdin.read()
   script_path.write_text(content, encoding="utf-8")
   __file__ = str(script_path)

ensure_venv_and_reexec()

# Imports
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
import uvicorn
import requests
import numpy as np
import networkx as nx
from bs4 import BeautifulSoup
from scipy.io import wavfile
from scipy.signal import stft

# Config
PORT = int(os.getenv("SC_PORT", "8767"))
HOST = os.getenv("SC_HOST", "0.0.0.0")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
TICK_SEC = float(os.getenv("SC_TICK_SEC", "1.0"))
REFLECT_EVERY = int(os.getenv("SC_REFLECT_EVERY", "5"))
DB_PATH = os.getenv("SC_DB_PATH", str(ROOT / "seed_crystal.db"))
SIGMA0 = float(os.getenv("SC_SIGMA0", "0.8"))
GAMMA = float(os.getenv("SC_GAMMA", "0.92"))
SIGMA_MIN = float(os.getenv("SC_SIGMA_MIN", "0.12"))
AUTONOMOUS_INGEST_EVERY = 20 # Ticks between autonomous ingests
X_SEARCH_QUERY = "(AI OR crystal OR sonification) filter:links min_faves:5" # Example for autonomous search

OUT_AUDIO = ROOT / "audio"; OUT_AUDIO.mkdir(parents=True, exist_ok=True)
OUT_SHAPES = ROOT / "shapes"; OUT_SHAPES.mkdir(parents=True, exist_ok=True)
OUT_STATE = ROOT / "state"; OUT_STATE.mkdir(parents=True, exist_ok=True)

# Utilities (from robust code)
def to_float32_pcm(x):

