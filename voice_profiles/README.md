# Voice Profiles Directory

This directory contains voice profile WAV files for voice cloning.

## Creating a Voice Profile

1. Record at least 5 seconds of clear speech
2. Save as WAV format (16-bit PCM, 16kHz preferred)
3. Use a quiet environment with no background noise
4. Speak naturally in your normal voice
5. Save the file as `my_voice.wav` or similar

## Using Your Voice Profile

Update `config.yaml` to point to your voice profile:

```yaml
enable_voice_clone: true
voice_profile_path: "./voice_profiles/my_voice.wav"
```

## Privacy Note

Voice profiles are personal data and should NOT be committed to git.
This directory is gitignored by default.
