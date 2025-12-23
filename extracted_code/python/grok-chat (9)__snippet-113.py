833
834 -    def _speak(self, text: str, *, input_rms: float, refractory_padding
     _s: float) -> None:
834 +    def _speak(self, text: str, *, input_rms: float, refractory_padding
     _s: float) -> tuple[float, float]:
835 +        """
836 +        Returns (processing_latency_ms, playback_latency_ms).
837 +        processing_latency_ms: time from entering this method to the st
     art of playback.
838 +        playback_latency_ms: approximate playback duration (blocking pl
     ay time).
839 +        """
840          if not SOUNDDEVICE_AVAILABLE:
836 -            return
841 +            return 0.0, 0.0
842 +
843 +        t_start = time.time()
844
    ⋮
899          elif self._tts_mode == "off":
893 -            return
900 +            return 0.0, 0.0
901          else:
902              # No TTS engine configured.
896 -            return
903 +            return 0.0, 0.0
904
905          if audio is None or audio.size == 0:
899 -            return
906 +            return 0.0, 0.0
907
    ⋮
915                  self._send_alert("mirror_gate_blocked", {"blocks": self
     ._gate_blocked, "reason": gate.reason})
909 -            return
916 +            return 0.0, 0.0
917          degraded = gate.degraded
    ⋮
922              self.stop()
916 -            return
923 +            return 0.0, 0.0
924
    ⋮
936          self._refractory_until = time.time() + duration_s + float(refra
     ctory_padding_s)
937 +        processing_latency_ms = float((time.time() - t_start) * 1000.0)
938          t_play_start = time.time()
    ⋮
940              sd.play(audio, samplerate=playback_sr, device=self._output_
     device, blocking=True)
933 -            self._last_playback_latency_ms = float((time.time() - t_pla
     y_start) * 1000.0)
941 +            playback_latency_ms = float((time.time() - t_play_start) *
     1000.0)
942          except Exception:
    ⋮
948              sd.play(audio_rs, samplerate=input_sr, device=self._output_
     device, blocking=True)
941 -            self._last_playback_latency_ms = float((time.time() - t_pla
     y_start) * 1000.0)
949 +            playback_latency_ms = float((time.time() - t_play_start) *
     1000.0)
950 +
951 +        return processing_latency_ms, playback_latency_ms
952

