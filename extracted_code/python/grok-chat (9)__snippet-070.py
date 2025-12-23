• The sherpa fallback to Vosk likely fails silently during Model import or initialization despite the model path existing, as
  indicated by missing expected Vosk logs and repeated sherpa errors; next, I'll verify the exact Vosk model path handed to
  Model and test Model loading directly to identify the error.

