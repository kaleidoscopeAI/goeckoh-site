`advanced_voice_mimic.py` has `VoiceCrystal` and `ProsodyProfile`. `echo_v4_complete` has a simpler `speak` function. I must UPGRADE `echo_v4_complete` to use the `VoiceCrystal` logic from `advanced_voice_mimic.py` because it's "advanced" and the user wants the "Voice Crystal" .
So: `Echo` class in `echo_v4_complete` needs to use `VoiceCrystal` instead of just `self.tts.tts_to_file`.

I will synthesize the `VoiceCrystal` class into the main script.

