1039                  sr = 16000
1040 -            stream = helper.create_stream()
1041 -            stream.accept_waveform(sr, data)
1042 -            helper.decode_stream(stream)
1043 -            result = helper.get_result(stream)
1044 -            text = result.text if hasattr(result, "text") else str(result)
1045 -            return jsonify({"text": text, "seconds": len(data) / sr})
1040 +            if self._asr_backend == "sherpa":
1041 +                stream = helper.create_stream()
1042 +                stream.accept_waveform(sr, data)
1043 +                helper.decode_stream(stream)
1044 +                result = helper.get_result(stream)
1045 +                text = result.text if hasattr(result, "text") else str(result)
1046 +            elif self._asr_backend == "vosk":
1047 +                from vosk import KaldiRecognizer  # type: ignore
1048 +                import json
1049 +                rec = KaldiRecognizer(helper, sr)
1050 +                rec.AcceptWaveform(data.tobytes())
1051 +                res = json.loads(rec.FinalResult())
1052 +                text = res.get("text", "")
1053 +            else:
1054 +                return jsonify({"error": "asr_backend_unknown"}), 500
1055 +            return jsonify({"text": text, "seconds": len(data) / sr, "backend": self._asr_backend})
1056          except Exception as e:

