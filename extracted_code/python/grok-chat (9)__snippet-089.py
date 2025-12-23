    1063                  import json
    1064 -                grammar = '["zero","one","two","three","four","five","six","seven","eight","nine"]'
    1065 -                rec = KaldiRecognizer(helper, sr, grammar)
    1064 +                grammar_env = os.environ.get("GOECKOH_ASR_GRAMMAR", "").lower()
    1065 +                grammar = None
    1066 +                if grammar_env in ("digits", "fsd", "numbers"):
    1067 +                    grammar = '["zero","one","two","three","four","five","six","seven","eight","nine"]'
    1068 +                rec = KaldiRecognizer(helper, sr, grammar) if grammar else KaldiRecognizer(helper, sr)
    1069                  pcm = (data * 32767.0).clip(-32768, 32767).astype("int16").tobytes()

