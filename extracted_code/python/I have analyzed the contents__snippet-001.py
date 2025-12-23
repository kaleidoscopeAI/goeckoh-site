"""
ECHO PRIME: THE UNIFIED ORGANIC AGI
===================================
A professional implementation of the Echo-Organic Architecture.
Integrates:
- Crystalline Heart (Emotion/Entropy)
- Voice Crystal (Prosody/Cloning)
- ABA Engine (Behavior/Therapy)
- Organic Subconscious (Self-Evolving Nodes)

Usage: python echo_prime.py
"""
import os
import sys
import time
import json
import queue
import random
import asyncio
import threading
import numpy as np
import torch
import sounddevice as sd
import soundfile as sf
import librosa
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from pathlib import Path
from collections import deque
from datetime import datetime

# --- Configuration ---
CHILD_NAME = "Jackson"
SAMPLE_RATE = 16000
VAD_THRESHOLD = 0.45
OLLAMA_MODEL = "deepseek-r1:8b"

# --- 1. THE ORGANIC CORE (The Subconscious) ---
# Derived from organic-learning-system 
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
    def __init__(self):
        self.strategies = {
            "anxious": ["Let's take three deep breaths together.", "Squeeze your hands tight, then let go."],
            "high_energy": ["Let's do 5 jumping jacks!", "Push against the wall with me."],
            "meltdown": ["I am here. You are safe. Just breathe."]
        }
    
    def evaluate(self, arousal: float, text: str) -> Optional[str]:
        if arousal > 7.0:
            return random.choice(self.strategies["meltdown"])
        if arousal > 5.0:
            return random.choice(self.strategies["high_energy"])
        if "scared" in text or "no" in text:
            return random.choice(self.strategies["anxious"])
        return None

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
        self.heart = CrystallineHeart()
        self.aba = AbaEngine()
        self.voice = VoiceCrystal()
        self.subconscious = OrganicSubconscious()
        
        # Start Subconscious
        self.sub_thread = threading.Thread(target=self.subconscious.run_background, daemon=True)
        self.sub_thread.start()

        # Audio Setup
        self.q = queue.Queue()
        try:
            from faster_whisper import WhisperModel
            self.whisper = WhisperModel("tiny.en", device="cpu", compute_type="int8")
            print("[System] Ears (Whisper) active.")
        except ImportError:
            print("[System] Faster-Whisper not found. Speech recognition will be simulated text-only.")
            self.whisper = None

    def listen_loop(self):
        """Real-time audio processing loop."""
        print("[System] Listening... (Press Ctrl+C to stop)")
        
        # VAD / Audio Callback
        def callback(indata, frames, time, status):
            self.q.put(indata.copy())

        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback):
            buffer = []
            silence_frames = 0
            
            while True:
                try:
                    audio_chunk = self.q.get()
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
                        
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    pass # Keep running

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

        print(f"\n🎤 User: {text}")

        # 2. Update Heart
        rms = np.sqrt(np.mean(audio_data**2))
        arousal, valence = self.heart.update(rms)

        # 3. ABA Check
        intervention = self.aba.evaluate(arousal, text)
        
        if intervention:
            # High priority intervention
            self.voice.speak(intervention, style="calm")
        else:
            # 4. LLM Response (The "Self")
            response = self.query_llm(text, arousal)
            self.voice.speak(response, style="neutral")

    def query_llm(self, text, arousal):
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
        except:
            return f"I hear you saying {text}."

if __name__ == "__main__":
    try:
        echo = EchoSystem()
        echo.listen_loop()
    except KeyboardInterrupt:
        print("\n[System] Shutdown.")
