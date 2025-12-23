  - Downloaded 10 WAV clips (digits 0–9, speaker “jackson”) from the public Free Spoken Digit Dataset into backend/data/free-
    spoken-digit/.
  - Included backend/data/free-spoken-digit/README.md with source and transcripts.

  You can use these immediately to sanity-check ASR/TTS/mirror latency without a GPU or large downloads. For example:

  # simple ASR sanity (using Sherpa if running)
  curl -F "file=@backend/data/free-spoken-digit/3_jackson_0.wav" http://localhost:8080/process

  If you want, I can wire a quick test script to batch-run these through /process or /mirror and report WER/latency.


