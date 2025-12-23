Always listening (VAD tuned for quiet/autistic speech – 1.2–1.8 s patience)
Child vocalizes → raw .wav captured
faster-whisper tiny.en → raw transcription
LanguageTool + custom autism articulation rules → corrected text (only clarity, never changes meaning)
VoiceCrystal (XTTS-v2 / Coqui with prosody transfer) → synthesizes corrected text using child’s exact voice + F0/energy contour from the original utterance
Plays back instantly (<1.8 s total latency on CPU) through headphones/private speaker
Child hears his own voice thinking the correct version → naturally repeats → loop reinforces
If child’s repeat matches → mark as mastery → harvest as new voice facet (slow-drift adaptation)

