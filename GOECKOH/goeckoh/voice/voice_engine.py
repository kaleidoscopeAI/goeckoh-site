import numpy as np
import os

try:
    from TTS.api import TTS
    NEURAL_TTS_AVAILABLE = True
except ImportError:
    print("[WARN] Coqui TTS not found. Neural voice cloning is disabled.")
    NEURAL_TTS_AVAILABLE = False
    TTS = None

class VoiceEngine:
    def __init__(self, use_gpu=False):
        self.use_neural = False
        self.model = None
        if NEURAL_TTS_AVAILABLE:
            try:
                # Try to load the best available model automatically
                available_models = [
                    "tts_models/en/vctk/vits",  # Multi-speaker VITS
                    "tts_models/en/ljspeech/vits",  # Single speaker fallback
                    "tts_models/en/ljspeech/tacotron2-DDC"  # Alternative model
                ]
                
                model_loaded = False
                for model_path in available_models:
                    try:
                        user_data_dir = TTS.get_user_data_dir()
                        model_dir = os.path.join(user_data_dir, model_path.replace('/', '--'))
                        
                        if not os.path.exists(model_dir):
                            print(f"[INFO] TTS model '{model_path}' not found, will be downloaded.")
                        
                        self.model = TTS(model_path, gpu=use_gpu)
                        self.use_neural = True
                        print(f"[INFO] VoiceEngine initialized with Coqui TTS model: {model_path}")
                        model_loaded = True
                        break
                    except Exception as e:
                        print(f"[WARN] Failed to load model {model_path}: {e}")
                        continue
                
                if not model_loaded:
                    print("[ERROR] All TTS models failed to load. Voice cloning disabled.")
                    self.use_neural = False
                    
            except Exception as e:
                print(f"[ERROR] Failed to initialize TTS: {e}. Voice cloning disabled.")
                self.use_neural = False

    def generate_speech_pcm(self, text: str, clone_ref_wav: str) -> np.ndarray | None:
        """Generate speech PCM using neural TTS with voice cloning"""
        if not self.use_neural or self.model is None:
            return None
            
        # Handle voice cloning if reference file exists
        if clone_ref_wav and os.path.exists(clone_ref_wav):
            try:
                # Try voice cloning with reference
                pcm_out = self.model.tts(text=text, speaker_wav=clone_ref_wav, language="en")
                return np.array(pcm_out, dtype=np.float32)
            except Exception as e:
                print(f"[WARN] Voice cloning failed: {e}. Falling back to regular synthesis.")
        
        # Fallback to regular synthesis without cloning
        try:
            pcm_out = self.model.tts(text=text, language="en")
            return np.array(pcm_out, dtype=np.float32)
        except Exception as e:
            print(f"[ERROR] TTS generation failed: {e}")
            return None
    
    def is_available(self) -> bool:
        """Check if neural TTS is available"""
        return self.use_neural and self.model is not None
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        if not self.is_available():
            return {"available": False}
        
        return {
            "available": True,
            "neural": True,
            "voice_cloning": True
        }
