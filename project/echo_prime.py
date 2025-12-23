"""ECHO PRIME: THE UNIFIED ORGANIC AGI
===================================
A professional implementation of the Echo-Organic Architecture.
Integrates:
- Crystalline Heart (Emotion/Entropy)
- Voice Crystal (Prosody/Cloning)
- ABA Engine (Behavior/Therapy)
- Organic Subconscious (Self-Evolving Nodes)

Usage: python echo_prime.py
"""
import base64 # Added for audio encoding
# import time # Already imported
import requests # Added for CCA backend communication
import os
import sys
import time
import json
import queue
import random
import asyncio
import threading
import numpy as np # Already imported
import torch
import sounddevice as sd
import soundfile as sf
import librosa
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable
from pathlib import Path
from collections import deque
from datetime import datetime

# --- Configuration ---
CHILD_NAME = "Jackson"
SAMPLE_RATE = 16000
VAD_THRESHOLD = 0.45
OLLAMA_MODEL = "deepseek-r1:8b"
CCA_BASE_URL = os.environ.get("CCA_BASE_URL", "http://127.0.0.1:5000") # Cognitive Crystal AI Backend URL
ECHO_DEVICE_ID = os.environ.get("ECHO_DEVICE_ID", "echo_local_01")
SESSION_ID = os.environ.get("SESSION_ID", "child_jackson_001")

class CCABridgeClient:
    def __init__(self, base_url: str, session_id: str, device_id: str):
        self.base_url = base_url.rstrip("/")
        self.session_id = session_id
        self.device_id = device_id

    def send_sensory_packet(self, packet: dict) -> None:
        url = f"{self.base_url}/api/sensory/packet"
        try:
            requests.post(url, json=packet, timeout=1.0)
        except Exception as e:
            print(f"[CCABridgeClient] Failed to send packet: {e}")

    def pull_commands(self, since_ts: float) -> list[dict]:
        url = f"{self.base_url}/api/commands/pull"
        try:
            r = requests.get(
                url,
                params={"session_id": self.session_id, "since": since_ts},
                timeout=1.0,
            )
            if r.status_code == 200 and r.content:
                return r.json()
        except Exception as e:
            print(f"[CCABridgeClient] Failed to pull commands: {e}")
        return []

# --- 1. THE ORGANIC CORE (The Subconscious) ---
# Derived from organic-learning-system 
# @.venv/lib/python3.12/site-packages/huggingface_hub/__pycache__/dataclasses.cpython-312.pyc
@dataclass
class NodeDNA:
    traits: List[float]
    def mutate(self) -> 'NodeDNA':
        # Evolution logic: gentle drift in traits
        new_traits = [t + random.uniform(-0.05, 0.05) for t in self.traits]
        return NodeDNA(traits=new_traits)

class OrganicNode:
    def __init__(self, node_id: str, dna: NodeDNA):
        self.id = node_id
        self.dna = dna
        self.energy = 1.0
        self.memory = []

    def metabolize(self, stimulus: float):
        # Grow or decay based on stimulus (entropy)
        self.energy += stimulus * self.dna.traits[0] # Trait 0 = Metabolism efficiency
        self.energy = max(0.0, min(10.0, self.energy))

    def replicate(self) -> Optional['OrganicNode']:
        if self.energy > 8.0:
            self.energy *= 0.5
            return OrganicNode(f"{self.id}_child", self.dna.mutate())
        return None

class OrganicSubconscious:
    """Background process that 'dreams' and grows based on system events."""
    def __init__(self):
        self.nodes = [OrganicNode("root", NodeDNA([0.1, 0.5, 0.9]))]
        self.lock = threading.Lock()
        self.running = True
    
    def feed(self, emotional_arousal: float):
        """Feed the organic brain with emotional data from the Heart."""
        with self.lock:
            for node in self.nodes:
                node.metabolize(emotional_arousal)
                child = node.replicate()
                if child:
                    self.nodes.append(child)
            # Prune dead nodes
            self.nodes = [n for n in self.nodes if n.energy > 0.1]

    def run_background(self):
        while self.running:
            time.sleep(1.0)
            # Simulate background processing ("dreaming")
            with self.lock:
                count = len(self.nodes)
                total_energy = sum(n.energy for n in self.nodes)
                # print(f"[Subconscious] Nodes: {count} | Energy: {total_energy:.2f}") # Debug

# --- 2. THE CRYSTALLINE HEART (Emotion Engine) ---
# Derived from agi_seed.py 
class CrystallineHeart:
    def __init__(self):
        self.arousal = 0.0
        self.valence = 0.0
        self.coherence = 1.0
        self.state_history = deque(maxlen=100)

    def update(self, audio_rms: float, text_sentiment: float = 0.0):
        # Simplified ODE for emotional dynamics
        decay = 0.95
        drive = audio_rms * 10.0
        
        self.arousal = (self.arousal * decay) + drive
        self.valence = (self.valence * 0.98) + text_sentiment
        
        # Coherence drops with high arousal (entropy increases)
        self.coherence = 1.0 / (1.0 + self.arousal * 0.1)
        
        self.state_history.append((self.arousal, self.valence))
        return self.arousal, self.valence

# --- 3. ABA ENGINE (The Therapist) ---
# Derived from aba/engine.py 
class AbaEngine:
    def __init__(self, voice_crystal): # Added voice_crystal argument
        self.strategies = {
            "anxious": "Let's take three deep breaths together.",
            "high_energy": "Let's do 5 jumping jacks!",
            "meltdown": "I am here. You are safe. Just breathe."
        }
        self.voice = voice_crystal # Store VoiceCrystal instance
    
    def evaluate(self, arousal: float, text: str) -> Optional[str]:
        if arousal > 7.0:
            return random.choice(self.strategies["meltdown"])
        if arousal > 5.0:
            return random.choice(self.strategies["high_energy"])
        if "scared" in text or "no" in text:
            return random.choice(self.strategies["anxious"])
        return None

    def apply_strategy(self, strategy_id: str, parameters: Dict[str, Any]) -> None:
        """Applies an ABA strategy, using VoiceCrystal to deliver feedback."""
        text_to_speak = parameters.get("text")
        if not text_to_speak:
            text_to_speak = self.strategies.get(strategy_id, "Strategy acknowledged.")
        
        self.voice.speak(text_to_speak, style=parameters.get("style", "calm"))

# --- 4. VOICE CRYSTAL (Speech Engine) ---
# Derived from advanced_voice_mimic.py 
class VoiceCrystal:
    def __init__(self):
        # Try to load Coqui TTS, fallback to pyttsx3 if missing/slow
        try:
            from TTS.api import TTS
            # Using a smaller, faster model for "production ready" responsiveness
            self.tts = TTS("tts_models/en/ljspeech/glow-tts", progress_bar=False, gpu=torch.cuda.is_available())
            self.engine_type = "neural"
            print("[VoiceCrystal] Neural TTS loaded.")
        except Exception as e:
            print(f"[VoiceCrystal] Neural TTS failed ({e}). Using standard TTS.")
            import pyttsx3
            self.engine = pyttsx3.init()
            self.engine_type = "standard"

    def speak(self, text: str, style: str = "neutral"):
        print(f"🗣️ [Echo ({style})]: {text}")
        if self.engine_type == "neural":
            # In a real scenario, we would apply style transfer here
            # For speed, we simply synthesize to a temp file and play
            self.tts.tts_to_file(text=text, file_path="output.wav")
            data, fs = sf.read("output.wav")
            sd.play(data, fs)
            sd.wait()
        else:
            self.engine.say(text)
            self.engine.runAndWait()

# --- 5. ECHO SYSTEM (The Orchestrator) ---
class EchoSystem:
    def __init__(self):
        print("[System] Initializing Echo Prime...")
        self.running = True # Control flag for the main loop
        self.heart = CrystallineHeart()
        self.voice = VoiceCrystal() # Initialize voice first
        self.aba = AbaEngine(self.voice) # Pass voice to ABA Engine
        self.subconscious = OrganicSubconscious()
        self.gui_callback: Optional[Callable[[str, Any], None]] = None
        
        # CCA Bridge Client
        self.turn_index = 0
        self._last_cmd_ts = 0.0
        self.cca_bridge = CCABridgeClient(CCA_BASE_URL, SESSION_ID, ECHO_DEVICE_ID)

        # Start Subconscious
        self.sub_thread = threading.Thread(target=self.subconscious.run_background, daemon=True)
        self.sub_thread.start()

        # Start CCA Command Poller
        self.start_command_loop()

        # Audio Setup
        self.q = queue.Queue()
        try:
            from faster_whisper import WhisperModel
            self.whisper = WhisperModel("tiny.en", device="cpu", compute_type="int8")
            self.log("LOG", "[System] Ears (Whisper) active.")
        except ImportError:
            print("[System] Faster-Whisper not found. Speech recognition will be simulated text-only.")
            self.whisper = None

    def start_command_loop(self):
        t = threading.Thread(target=self._command_loop, daemon=True)
        t.start()

    def _command_loop(self):
        while self.running: # Use self.running flag
            cmds = self.cca_bridge.pull_commands(self._last_cmd_ts)
            for cmd in cmds:
                self._last_cmd_ts = max(self._last_cmd_ts, float(cmd.get("ts", 0.0)))
                self.handle_cca_command(cmd)
            time.sleep(0.25)

    def handle_cca_command(self, cmd: dict) -> None:
        for action in cmd.get("actions", []):
            a_type = action.get("type")
            if a_type == "speech":
                text = action.get("text", "")
                prosody = action.get("prosody") or {}
                self.voice.speak(
                    text=text,
                    # Placeholder for actual prosody parameters in voice.speak
                    # You'd need to adapt voice.speak to accept these
                    style=prosody.get("style", "neutral"), # Default style
                )
            elif a_type == "aba":
                strategy_id = action.get("strategy_id", "default")
                parameters = action.get("parameters", {})
                self.log("ABA_STRATEGY", f"CCA triggered ABA: {strategy_id} with {parameters}")
                self.aba.apply_strategy(strategy_id, parameters) # Call apply_strategy
            else:
                self.log("ERROR", f"Unknown CCA command type: {a_type} with value: {action}")
            
    def set_gui_callback(self, callback_func: Callable[[str, Any], None]):
        """Allows the GUI to hook into the system logs."""
        self.gui_callback = callback_func

    def log(self, message_type: str, content: Any):
        """Sends data to GUI if attached, else prints."""
        if self.gui_callback:
            self.gui_callback(message_type, content)
        else:
            print(f"[{message_type}] {content}")

    def listen_loop(self):
        """Real-time audio processing loop."""
        self.log("LOG", "[System] Listening... (Press Ctrl+C to stop)")
        
        # VAD / Audio Callback
        def callback(indata, frames, time, status):
            self.q.put(indata.copy())

        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback):
            buffer = []
            silence_frames = 0
            
            while self.running: # Use self.running flag
                try:
                    audio_chunk = self.q.get(timeout=1.0) # Add a timeout to check self.running
                    rms = np.sqrt(np.mean(audio_chunk**2))
                    
                    # Feed the Organic Subconscious directly
                    self.subconscious.feed(rms * 100) # Amplify small signals
                    
                    if rms > 0.01: # Speech detected
                        buffer.append(audio_chunk)
                        silence_frames = 0
                    else:
                        silence_frames += 1

                    # End of utterance detection (approx 1.2s silence)
                    if silence_frames > 20 and buffer: 
                        audio_full = np.concatenate(buffer)
                        self.process_utterance(audio_full)
                        buffer = [] # Reset
                        
                except queue.Empty: # Handle timeout, check self.running
                    continue
                except KeyboardInterrupt:
                    self.log("LOG", "\n[System] KeyboardInterrupt detected. Shutting down.")
                    self.running = False
                except Exception as e:
                    self.log("ERROR", f"Error in listen_loop: {e}")
                    # Keep running, but log the error
                    
        self.log("LOG", "[System] Listen loop terminated.")
        self.subconscious.running = False # Stop subconscious thread

    def process_utterance(self, audio_data):
        """The Cognitive Pipeline."""
        # 1. Transcribe
        text = ""
        if self.whisper:
            # Convert to float32 for Whisper
            audio_float = audio_data.flatten().astype(np.float32)
            segments, _ = self.whisper.transcribe(audio_float, beam_size=5)
            text = " ".join([s.text for s in segments])
        
        if not text.strip(): return

        self.log("TRANSCRIPT", f"🎤 User: {text}")

        # 2. Update Heart
        rms = np.sqrt(np.mean(audio_data**2))
        arousal, valence = self.heart.update(rms)
        self.log("INTENSITY", arousal)

        # --- NEW: Send sensory packet to CCA Backend ---
        self.turn_index += 1

        packet = {
            "session_id": SESSION_ID,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
            "speaker": "child",
            "raw_text": text,
            "clean_text": text, # For now, clean_text is same as raw_text
            "language": "en",
            "emotion": {
                "arousal": float(arousal),
                "valence": float(valence),
                "confidence": 1.0, # Placeholder
            },
            "audio": {
                "sample_rate": int(SAMPLE_RATE),
                "format": "wav16",
                "base64": self._encode_audio(audio_data, SAMPLE_RATE),
            },
            "aba": {
                "skill_tag": "default", # Placeholder
                "prompt_level": "none", # Placeholder
                "reinforcement_tag": "none", # Placeholder
                "behavior_notes": "none", # Placeholder
            },
            "echo_meta": {
                "utterance_id": f"echo_utt_{self.turn_index:06d}",
                "turn_index": int(self.turn_index),
                "device_id": ECHO_DEVICE_ID,
            },
        }

        self.cca_bridge.send_sensory_packet(packet)
        # --- END NEW ---

        # The ABA Check and LLM Response will now be handled by CCA backend commands
        # Therefore, we remove the local ABA/LLM logic here and wait for CCA commands.
        # Original ABA Check and LLM Response logic removed.


    def query_llm(self, text: str, arousal: float) -> str:
        """Get a response from the local LLM personality."""
        try:
            import ollama
            prompt = f"""
            You are Echo, a supportive inner voice for an autistic child.
            The child just said: "{text}".
            Their arousal level is {arousal:.2f} (0-10).
            Reply in 1 short sentence. Be kind, validating, and use 'I' statements as if you are their inner voice.
            """
            res = ollama.generate(model=OLLAMA_MODEL, prompt=prompt)
            return res['response']
        except Exception as e:
            self.log("ERROR", f"LLM Query Error: {e}")
            return f"I hear you saying {text}."

    def _encode_audio(self, audio_buffer: np.ndarray, sr: int) -> str:
        # audio_buffer assumed float32 -1..1
        pcm16 = (audio_buffer * 32767.0).clip(-32768, 32767).astype("<i2")
        return base64.b64encode(pcm16.tobytes()).decode("ascii")

if __name__ == "__main__":
    try:
        echo = EchoSystem()
        echo.listen_loop()
    except KeyboardInterrupt:
        print("\n[System] Shutdown initiated from main block.")
    except Exception as e:
        print(f"\n[System] Critical error in main: {e}")
    finally:
        if 'echo' in locals() and echo.running:
            echo.running = False
            # Give background threads a moment to shut down
            time.sleep(2)
            print("[System] EchoSystem gracefully shut down.")
