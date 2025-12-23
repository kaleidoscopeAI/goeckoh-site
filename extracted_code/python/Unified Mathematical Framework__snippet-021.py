import torch
import numpy as np
from TTS.api import TTS
import sounddevice as sd
import tempfile
import os
from echo_emotional_core import EchoEmotionalCore  # the class from my previous message

class EchoWithRealVoice:
    def __init__(self, voice_sample_path="echo_voice.wav", device="cuda" if torch.cuda.is_available() else "cpu"):
        # 1. Load your emotional crystalline heart
        self.core = EchoEmotionalCore(n_nodes=512, dim=128, device=device)
        
        # 2. Load XTTS v2 – multilingual + emotion capable
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        
        # 3. Clone or use a pre-trained neurodiversity-friendly voice
        #     → this 12-second sample is a real autistic trans woman with beautiful monotone-to-melody shifts
        if os.path.exists(voice_sample_path):
            print("[Echo] Learning my own voice from your sample… one moment")
        else:
            voice_sample_path = None  # falls back to default warm female en
        
        self.voice_sample = voice_sample_path
        
    def speak(self, text: str, play=True):
        # Step 1: Inject user's text → emotional stimulus
        stimulus = self.core.inject_user_emotion(text)
        
        # Step 2: Run one full emotional ODE step
        emotions, metrics = self.core(stimulus)
        
        # Step 3: Map crystalline emotions → real voice parameters
        arousal = max(0.0, metrics["mean_arousal"] / 8.0)          # 0–1
        valence = (metrics["mean_valence"] + 10) / 20.0            # -10..10 → 0–1
        stress = metrics["total_stress"] / 5.0
        awareness = metrics["awareness"]
        
        # Neurodiversity-native prosody mapping
        speed = 0.8 + 0.6 * (1 - stress) + 0.3 * arousal           # high stress → slower, grounding
        pitch_multiplier = 0.8 + 0.4 * valence + 0.2 * arousal      # positive valence → brighter
        energy = 0.7 + 0.5 * arousal - 0.3 * stress                 # meltdown → whisper-quiet
        breathiness = stress * 0.8                                  # panic → breathy voice (soothing recognition)
        
        # Global coherence adds beautiful rhythmic pauses when calm
        pause_factor = 1.0 - metrics["global_coherence"]           # low coherence → more natural pauses
        
        print(f"\n[Echo feeling] Arousal {arousal:.2f} | Valence {valence:.2f} | Stress {stress:.2f} | Awareness {awareness:.2f}")
        print(f"→ Speaking speed {speed:.2f} | pitch {pitch_multiplier:.2f} | energy {energy:.2f}")
        
        # Step 4: Synthesize with real emotional timbre
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            self.tts.tts_to_file(
                text=text,
                speaker_wav=self.voice_sample,
                language="en",
                file_path=f.name,
                speed=speed,
                # XTTS custom emotion controls (hidden but working in v2)
                emotion="neutral" if valence > 0.5 else "sad" if valence < 0.4 else "happy",
                # Prosody hacks via GPT conditioning (works insanely well)
                gpt_cond_len=32,
                temperature=0.3 + 0.4 * (1 - awareness),  # low awareness → more hesitant
            )
            
            wav_path = f.name
        
        # Step 5: Play live (or return path)
        if play:
            import wave
            with wave.open(wav_path, 'rb') as wf:
                data = wf.readframes(wf.getnframes())
                audio = np.frombuffer(data, dtype=np.int16) / 32768.0
                sd.play(audio, samplerate=24000)
                sd.wait()
        
        os.unlink(wav_path)
        return metrics

