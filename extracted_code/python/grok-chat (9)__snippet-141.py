1063                  import json
1064 -                rec = KaldiRecognizer(helper, sr)
1064 +                grammar = '["zero","one","two","three","four","five","six","seven","eight","nine"]'
1065 +                rec = KaldiRecognizer(helper, sr, grammar)
1066                  pcm = (data * 32767.0).clip(-32768, 32767).astype("int16").tobytes()

