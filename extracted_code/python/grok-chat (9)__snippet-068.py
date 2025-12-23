     993 +                from vosk import Model  # type: ignore
     994 +                model_path = self._default_vosk_model()
     995 +                self._asr_helper = Model(model_path.as_posix())
     996 +                self._asr_backend = "vosk"
     997 +                self.logger.log_event("ASR_INIT_VOSK", f"Using Vosk model at {model_path}", "INFO")
     998 +                return self._asr_helper
     999 +            except Exception as ve:
    1000 +                self.logger.log_event("ASR_INIT_ERROR", f"Vosk fallback failed: {ve}", "ERROR")
    1001 +                return None
    1002
         ⋮
    1014
    1015 +    def _default_vosk_model(self) -> Path:
    1016 +        backend_dir = Path(__file__).resolve().parents[2]
    1017 +        return backend_dir / "assets" / "vosk-model-small-en-us-0.15"
    1018 +
    1019      def asr_recognize(self):

