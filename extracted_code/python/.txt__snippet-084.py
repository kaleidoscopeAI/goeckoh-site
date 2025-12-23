from __future__ import annotations
import base64
import csv
import io
import math
import os
import queue
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import sounddevice as sd
import wave
import pyaudio
from flask import Flask, jsonify, render_template_string, request
