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

