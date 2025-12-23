    973 +                from vosk import Model  # type: ignore
    974 +                model_path = self._default_vosk_model()
    975 +                self._asr_helper = Model(model_path.as_posix())
    976 +                self._asr_backend = "vosk"
    977 +                self.logger.log_event("ASR_INIT_VOSK", f"Using Vosk model at {model_path}", "INFO")
    978 +                return self._asr_helper
    979 +            except Exception as ve:
    980 +                self.logger.log_event("ASR_INIT_ERROR", f"Vosk init failed: {ve}", "ERROR")
    981 +                if preferred == "vosk":
    982 +                    return None
    983          try:

