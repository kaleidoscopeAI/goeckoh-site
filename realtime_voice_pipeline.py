"""
Goeckoh Real-Time Voice Pipeline (v4 - Full-Stack Integration)
============================================================

This script acts as the "glue" for the Goeckoh system, creating a single,
runnable pipeline that realizes the primary, non-negotiable goal:
a low-latency, offline, voice-to-clone feedback loop with synchronized
visuals.

**ENHANCEMENT (V4):** This version integrates the WebSocket `BubbleBroadcaster`
to connect the backend audio pipeline with the `bubble_viewer.html` frontend.
The system now provides both audio and visual feedback in real-time.

**Pipeline States:**
1. **Intelligent Enrollment Mode**: Captures and phonetically analyzes utterances
   to build a voice profile. Provides basic visual feedback for any sound.
2. **Active Cloning Mode**: Performs the full STT -> Correction -> TTS ->
   Prosody Transfer -> Playback loop, while simultaneously driving the 3D
   bubble visualization.

**To Run:**
1. Ensure all dependencies are installed.
2. Open `goeckoh_cloner/webui/bubble_viewer.html` in a browser.
3. Run this script from the project root, specifying a profile name:
   `python realtime_voice_pipeline.py --profile-name <your_profile_name>`
"""

import argparse
import os
import queue
import sys
import threading
import time
import tempfile
import uuid
import json
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
import webrtcvad
import librosa

# --- Correcting Python Paths & Importing Components ---
sys.path.insert(0, '.')

# Core Pipeline
from goeckoh_cloner.goeckoh_loop import AudioCapture, VADUtteranceDetector
from src.neuro_backend import NeuroKernel
from goeckoh_cloner.correction_engine import clean_asr_text
from pipeline_core.tts import synthesize_text_to_wav
from psychoacoustic_engine.voice_profile import VoiceFingerprint, get_profile_path
from goeckoh.voice.prosody import extract_prosody_from_int16, apply_prosody_to_tts
from psychoacoustic_engine.voice_logger import log_voice_characteristics
from goeckoh_cloner.config import PROFILES_DIR

# Visual Pipeline
from goeckoh_cloner.phoneme_utils import text_to_phoneme_classes, PhonemeClass
from goeckoh_cloner.visual_engine_bridge import BubbleBroadcaster, compute_bubble_state
from psychoacoustic_engine.bubble_synthesizer import feed_text_through_bubble
from psychoacoustic_engine.attempt_analysis import analyze_chunk

# --- Intelligent Enrollment Component ---

class PhonemeCoverageTracker:
    # ... (omitted for brevity, same as v3) ...
    """Tracks the collected duration for each phoneme class."""
    def __init__(self, profile_name: str):
        self.path = PROFILES_DIR / f"{profile_name}_phoneme_tracker.json"
        self.coverage = {pc.name: 0.0 for pc in PhonemeClass}
        self.load()

    def load(self):
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as f:
                self.coverage.update(json.load(f))

    def save(self):
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(self.coverage, f, indent=2)

    def add_snippet(self, phoneme_classes: list, duration: float):
        if not phoneme_classes:
            return
        duration_per_class = duration / len(phoneme_classes)
        for pc in phoneme_classes:
            self.coverage[pc.name] += duration_per_class
    
    def get_coverage_summary(self, required_duration: float) -> str:
        lines = []
        for pc_name, duration in self.coverage.items():
            progress = min(1.0, duration / required_duration) * 100
            lines.append(f"   - {pc_name:<12}: {duration:.2f}s / {required_duration:.2f}s ({progress:.0f}%)")
        return "\n".join(lines)

    def is_coverage_sufficient(self, required_duration_per_class: float) -> bool:
        num_sufficient = sum(1 for d in self.coverage.values() if d >= required_duration_per_class)
        return num_sufficient >= len(self.coverage) * 0.9

# --- Main Pipeline Orchestrator ---

class RealtimeVoicePipeline:
    def __init__(self, profile_name: str, required_enrollment_sec: float = 1.5):
        print("✅ Initializing Goeckoh Real-Time Pipeline (v4 - Full Stack)...")
        self.profile_name = profile_name
        self.stop_event = threading.Event()
        
        self.required_duration_per_class = required_enrollment_sec
        self.enrollment_cache_dir = PROFILES_DIR / "enrollment_cache" / self.profile_name
        self.enrollment_cache_dir.mkdir(parents=True, exist_ok=True)

        self.profile_path = get_profile_path(self.profile_name)
        self.clone_ref_wav_path = PROFILES_DIR / "refs" / f"{self.profile_name}_ref.wav"
        
        self.is_enrolled = self.profile_path.exists() and self.clone_ref_wav_path.exists()
        
        if self.is_enrolled:
            self.voice_fingerprint = VoiceFingerprint.from_json(self.profile_path)
            print(f"🧬 Loaded VoiceFingerprint for '{self.profile_name}'.")

        # Initialize Visual Engine Bridge
        print("🌍 Initializing WebSocket Bridge for Bubble Visualization...")
        self.ws_broadcaster = BubbleBroadcaster()

        # Initialize Audio & AI Components
        print("🎤 Initializing Audio Capture and VAD...")
        self.capture = AudioCapture()
        self.vad = VADUtteranceDetector()

        print("📝 Initializing STT Engine (sherpa-onnx)...")
        try:
            self.asr = NeuroKernel().asr
        except Exception as e:
            raise RuntimeError(f"Failed to load sherpa-onnx ASR models. Error: {e}")
        
        self.phoneme_tracker = PhonemeCoverageTracker(self.profile_name)
        
        self.playback_queue = queue.Queue()
        self.playback_thread = threading.Thread(target=self._playback_worker, daemon=True)

    def _playback_worker(self):
        # ... (omitted for brevity, same as v3) ...
        while not self.stop_event.is_set():
            try:
                audio_data, sr = self.playback_queue.get(timeout=1)
                sd.play(audio_data, samplerate=sr)
                sd.wait()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Error] Audio playback failed: {e}")

    def _enroll_from_snippets(self):
        # ... (omitted for brevity, same as v3) ...
        print("⏳ Starting enrollment process from collected audio snippets...")
        audio_files = list(self.enrollment_cache_dir.glob("*.wav"))
        if not audio_files: return False
        try:
            audio_samples = [sf.read(f, dtype='float32')[0] for f in audio_files]
            log_voice_characteristics(
                audio_samples=audio_samples, sr=self.capture.sample_rate, user_id=self.profile_name,
                output_dir=PROFILES_DIR.parent, speaker_embedding=None
            )
            self.is_enrolled = True
            self.voice_fingerprint = VoiceFingerprint.from_json(self.profile_path) # Load new profile
            for f in audio_files: f.unlink()
            print("✅ Enrollment successful! Active Cloning Mode is now enabled.")
            return True
        except Exception as e:
            print(f"❌ Enrollment process failed: {e}")
            return False

    def _run_bubble_animation(self, control_curves: dict):
        """Animates the bubble based on generated control curves."""
        t = control_curves["t"]
        n = len(t)
        for i in range(n):
            if self.stop_event.is_set():
                break
            # Note: The original `compute_bubble_state` uses a `VoiceFingerprint` object
            state = compute_bubble_state(self.voice_fingerprint, control_curves, i, idle=False)
            self.ws_broadcaster.send_state(state)
            if i < n - 1:
                dt = t[i + 1] - t[i]
                time.sleep(max(dt, 0.005))
    
    def _run_raw_audio_bubble(self, audio_chunk: np.ndarray):
        """Provides direct visual feedback for any sound during enrollment."""
        try:
            # Use the simpler feature extractor for real-time feedback
            features = analyze_chunk(audio_chunk, sr=self.capture.sample_rate)
            num_frames = len(features.energy_attempt)
            for i in range(num_frames):
                # We need a dummy profile for bubble_foam, so create a default one
                dummy_fp = VoiceFingerprint()
                state = {
                    "radius": 1.0 + features.energy_attempt[i] * 2.0,
                    "spike": features.zcr_attempt[i] * 2.0,
                    "metalness": features.spectral_tilt[i],
                    "roughness": 1.0 - features.hnr_attempt[i],
                    "hue": (np.clip(features.f0_attempt[i], 80, 400) - 80) / 320.0,
                    "halo": features.energy_attempt[i],
                    "idle": False,
                }
                self.ws_broadcaster.send_state({k: np.clip(v, 0, 1) if k != 'radius' else v for k, v in state.items()})
                time.sleep(features.dt)
        except Exception as e:
            print(f"Raw audio visualization failed: {e}")

    def run(self):
        """Starts the main pipeline."""
        self.capture.start()
        self.playback_thread.start()
        
        if self.is_enrolled:
            print("\n✅ Profile found. Starting in Active Cloning Mode.")
            self._active_cloning_loop()
        else:
            print("\n⏳ No profile found. Starting in Intelligent Passive Enrollment Mode.")
            self._intelligent_enrollment_loop()

    def _intelligent_enrollment_loop(self):
        # ... (omitted for brevity, mostly same as v3, with visualization added) ...
        print("   - Goal: Collect at least " f"{self.required_duration_per_class:.1f}s for each major phoneme class.")
        print("   - Current Progress:")
        print(self.phoneme_tracker.get_coverage_summary(self.required_duration_per_class))
        
        while not self.stop_event.is_set() and not self.is_enrolled:
            utterance_audio_float32 = self.vad.detect_utterance(self.capture, self.stop_event)
            if utterance_audio_float32 is None: continue

            # --- NEW: Provide immediate visual feedback for any sound ---
            threading.Thread(target=self._run_raw_audio_bubble, args=(utterance_audio_float32,), daemon=True).start()

            stream = self.asr.create_stream()
            stream.accept_waveform(self.capture.sample_rate, utterance_audio_float32)
            self.asr.decode_stream(stream)
            raw_text = self.asr.get_result(stream).strip()
            
            if not raw_text: continue

            phoneme_classes = text_to_phoneme_classes(raw_text)
            phoneme_class_names = {pc.name for pc in phoneme_classes}
            is_useful = any(self.phoneme_tracker.coverage[pc_name] < self.required_duration_per_class for pc_name in phoneme_class_names)

            if is_useful:
                snippet_duration = len(utterance_audio_float32) / self.capture.sample_rate
                snippet_path = self.enrollment_cache_dir / f"snippet_{int(time.time() * 1000)}.wav"
                sf.write(snippet_path, utterance_audio_float32, self.capture.sample_rate)
                self.phoneme_tracker.add_snippet(phoneme_classes, snippet_duration)
                self.phoneme_tracker.save()
                
                print(f"\n✅ Useful snippet captured ({snippet_duration:.2f}s). Contains: {', '.join(phoneme_class_names)}")
                print("   - Updated Progress:")
                print(self.phoneme_tracker.get_coverage_summary(self.required_duration_per_class))

                if self.phoneme_tracker.is_coverage_sufficient(self.required_duration_per_class):
                    if self._enroll_from_snippets():
                        self._active_cloning_loop()
                        return
            else:
                print("   - Snippet ignored (sounds already well-represented).")

    def _active_cloning_loop(self):
        # ... (omitted for brevity, same as v3, with visualization added) ...
        while not self.stop_event.is_set():
            print("\n✅ Listening for speech...")
            utterance_audio_float32 = self.vad.detect_utterance(self.capture, self.stop_event)
            if utterance_audio_float32 is None: continue
            
            t0 = time.time()
            utterance_audio_int16 = (utterance_audio_float32 * 32767).astype(np.int16)
            
            print("...Processing utterance...")
            stream = self.asr.create_stream()
            stream.accept_waveform(self.capture.sample_rate, utterance_audio_float32)
            self.asr.decode_stream(stream)
            raw_text = self.asr.get_result(stream).strip()
            print(f"   - Raw Text: '{raw_text}'")

            if not raw_text: continue

            corrected_text = clean_asr_text(raw_text)
            print(f"   - Corrected Text: '{corrected_text}'")
            
            # --- NEW: Generate visual control curves and start animation ---
            control_curves = feed_text_through_bubble(corrected_text, self.voice_fingerprint)
            anim_thread = threading.Thread(target=self._run_bubble_animation, args=(control_curves,), daemon=True)
            anim_thread.start()

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_synth_file:
                synth_path = tmp_synth_file.name
            
            synthesize_text_to_wav(
                corrected_text, out_wav=synth_path, voice_profile_wav=str(self.clone_ref_wav_path),
                use_voice_clone=True
            )
            
            original_prosody = extract_prosody_from_int16(utterance_audio_int16, self.capture.sample_rate)
            if original_prosody:
                print("   - Applying original prosody to cloned voice...")
                apply_prosody_to_tts(Path(synth_path), original_prosody)

            final_audio, final_sr = sf.read(synth_path, dtype='float32')
            self.playback_queue.put((final_audio, final_sr))
            
            try: os.remove(synth_path)
            except OSError as e: print(f"Error removing temp file {synth_path}: {e}")

            latency = (time.time() - t0) * 1000
            print(f"   - Pipeline Latency: {latency:.2f} ms")

            if anim_thread.is_alive():
                anim_thread.join()

    def shutdown(self):
        # ... (omitted for brevity, same as v3) ...
        print("Shutting down pipeline...")
        self.stop_event.set()
        self.capture.stop()
        if self.playback_thread.is_alive():
            self.playback_thread.join(timeout=2)
        print("✅ Pipeline stopped.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Goeckoh Real-Time Voice-to-Clone Pipeline.")
    parser.add_argument(
        "--profile-name", required=True,
        help="The name for the voice profile. If it doesn't exist, it will be created via passive enrollment."
    )
    parser.add_argument(
        "--enrollment-sec-per-class", type=float, default=1.5,
        help="Target seconds of audio to collect for each phoneme class."
    )
    args = parser.parse_args()
    
    pipeline = None
    try:
        pipeline = RealtimeVoicePipeline(
            profile_name=args.profile_name,
            required_enrollment_sec=args.enrollment_sec_per_class
        )
        pipeline.run()
    except KeyboardInterrupt:
        print("\n🛑 Stop signal received.")
    except (FileNotFoundError, RuntimeError, ImportError) as e:
        print(f"\n❌ FATAL ERROR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        if pipeline:
            pipeline.shutdown()