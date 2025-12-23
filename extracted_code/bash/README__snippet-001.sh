# Record 5s sample and use for cloning
python -m cli speak --record --duration 5 --voice-profile ./sample_voice.wav

# Use existing WAV sample
python -m cli speak --input-file ./input.wav --voice-profile ./sample_voice.wav
