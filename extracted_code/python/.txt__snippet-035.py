from __future__ import annotations
import os
import math
import time
import queue
import threading
import tempfile
import hashlib
import struct
import re
import csv
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import sounddevice as sd
import wave
import pyaudio
from flask import Flask, jsonify, render_template_string
