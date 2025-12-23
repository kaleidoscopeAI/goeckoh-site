     802 +    def _next_pow2(n: int) -> int:
     803 +        n = int(n)
     804 +        if n <= 1:
     805 +            return 1
     806 +        return 1 << (n - 1).bit_length()
     807 +
     808 +    def _reset_denoise_state(self) -> None:
     809 +        """Reset VAD/noise-reduction state for a new mirror session.""
          "
     810 +        block_ms = 1000.0 * float(self._blocksize) / max(1.0, float(se
          lf._sample_rate))
     811 +        self._vad_hangover_frames = max(0, int(round(self._vad_hangove
          r_ms / max(1.0, block_ms))))
     812 +        self._vad_hangover = 0
     813 +        preroll_frames = max(0, int(round(self._vad_preroll_ms / max(1
          .0, block_ms))))
     814 +        self._preroll = deque(maxlen=preroll_frames)
     815 +        self._vad_active = False
     816 +        self._last_vad_threshold = self._noise_floor * self._vad_margi
          n
     817 +        self._noise_psd = None
     818 +        self._last_snr_db = 0.0
     819 +        self._snr_hist.clear()
     820 +
     821 +        if self._nr_enabled:
     822 +            fft_size = max(self._nr_fft_size, self._blocksize)
     823 +            self._nr_fft_size = self._next_pow2(fft_size)
     824 +            self._nr_window = np.hanning(self._nr_fft_size).astype(np.
          float32)
     825 +            gain = float(np.mean(self._nr_window)) if self._nr_window.
          size else 1.0
     826 +            self._nr_window_gain = gain if gain > 1e-6 else 1.0
     827 +        else:
     828 +            self._nr_window = None
     829 +            self._nr_window_gain = 1.0
     830 +
     831 +    def _compute_vad_threshold(self) -> float:
     832 +        thr = float(self._noise_floor) * float(self._vad_margin)
     833 +        thr = max(self._vad_min_floor, min(thr, self._vad_max_floor))
     834 +        self._last_vad_threshold = float(thr)
     835 +        return float(thr)
     836 +
     837 +    def _estimate_snr_db(self, rms: float) -> float:
     838 +        if rms <= 1e-8:
     839 +            return -80.0
     840 +        floor = max(float(self._noise_floor), 1e-6)
     841 +        snr_db = 20.0 * float(np.log10(max(rms / floor, 1e-8)))
     842 +        return float(np.clip(snr_db, -20.0, 60.0))
     843 +
     844 +    def _update_noise_floor(self, rms: float) -> None:
     845 +        if self._noise_adapt <= 0.0:
     846 +            return
     847 +        a = float(np.clip(self._noise_adapt, 0.0, 1.0))
     848 +        self._noise_floor = max(self._vad_min_floor, (1.0 - a) * self.
          _noise_floor + a * float(rms))
     849 +
     850 +    def _vad_is_speech(self, rms: float, snr_db: float) -> bool:
     851 +        if not self._vad_enabled:
     852 +            return rms >= float(self._noise_floor)
     853 +        thr = self._compute_vad_threshold()
     854 +        if rms < thr:
     855 +            return False
     856 +        if snr_db < float(self._vad_snr_db):
     857 +            return False
     858 +        return True
     859 +
     860 +    def _rfft_chunk(self, x: np.ndarray) -> np.ndarray:
     861 +        n = int(self._nr_fft_size)
     862 +        if n <= 0:
     863 +            n = int(x.size)
     864 +        if x.size < n:
     865 +            pad = np.zeros(n, dtype=np.float32)
     866 +            pad[: x.size] = x.astype(np.float32, copy=False)
     867 +        else:
     868 +            pad = x[:n].astype(np.float32, copy=False)
     869 +        if self._nr_window is not None:
     870 +            pad = pad * self._nr_window
     871 +        return np.fft.rfft(pad)
     872 +
     873 +    def _update_noise_profile(self, x: np.ndarray) -> None:
     874 +        if not self._nr_enabled:
     875 +            return
     876 +        if x.size == 0:
     877 +            return
     878 +        spec = self._rfft_chunk(x)
     879 +        psd = (np.abs(spec) ** 2).astype(np.float32)
     880 +        if self._noise_psd is None or self._noise_psd.shape != psd.sha
          pe:
     881 +            self._noise_psd = psd
     882 +            return
     883 +        a = float(np.clip(self._nr_alpha, 0.0, 1.0))
     884 +        self._noise_psd = (1.0 - a) * self._noise_psd + a * psd
     885 +
     886 +    def _denoise_chunk(self, x: np.ndarray) -> np.ndarray:
     887 +        if not self._nr_enabled or x.size == 0:
     888 +            return x
     889 +        if self._noise_psd is None:
     890 +            return x
     891 +
     892 +        spec = self._rfft_chunk(x)
     893 +        psd = (np.abs(spec) ** 2).astype(np.float32)
     894 +        if self._noise_psd.shape != psd.shape:
     895 +            self._noise_psd = psd
     896 +            return x
     897 +
     898 +        use_wiener = self._nr_mode in ("wiener", "both", "wiener+gate"
          , "wiener_gate")
     899 +        use_gate = self._nr_mode in ("spectral_gate", "gate", "both",
          "wiener+gate", "wiener_gate")
     900 +        if not use_wiener and not use_gate:
     901 +            return x
     902 +
     903 +        if use_wiener:
     904 +            gain = 1.0 - (self._noise_psd / (psd + 1e-8))
     905 +            gain = np.clip(gain, float(self._nr_min_gain), 1.0)
     906 +        else:
     907 +            gain = np.ones_like(psd, dtype=np.float32)
     908 +
     909 +        if use_gate:
     910 +            mag = np.sqrt(psd)
     911 +            noise_mag = np.sqrt(self._noise_psd)
     912 +            gate = (mag >= (noise_mag * float(self._nr_gate_factor))).
          astype(np.float32)
     913 +            gain = gain * gate
     914 +
     915 +        spec = spec * gain
     916 +        y = np.fft.irfft(spec, n=self._nr_fft_size).astype(np.float32,
           copy=False)
     917 +        if self._nr_window is not None and self._nr_window_gain > 1e-6
          :
     918 +            y = y / float(self._nr_window_gain)
     919 +        if y.size > x.size:
     920 +            y = y[: x.size]
     921 +        elif y.size < x.size:
     922 +            y = np.pad(y, (0, x.size - y.size), mode="constant")
     923 +        return y.astype(np.float32, copy=False)
     924 +
     925      def _run_loop(self, *, sensitivity: float, refractory_padding_s: f
          loat) -> None:
         ⋮
     941      def _handle_audio_chunk(self, chunk: np.ndarray, *, sensitivity: f
          loat, refractory_padding_s: float) -> None:
     942 +        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
     943          if chunk.size == 0:
     944              return
     945 +        if not np.all(np.isfinite(chunk)):
     946 +            chunk = np.nan_to_num(chunk, copy=False)
     947
         ⋮
     955                  self._log.info("SpeechMirror calibrated noise floor: %
          .4f", self._noise_floor)
     956 +            if self._nr_enabled:
     957 +                self._update_noise_profile(chunk)
     958              return
     959
     831 -        if rms < self._noise_floor:
     960 +        snr_db = self._estimate_snr_db(rms)
     961 +        self._last_snr_db = float(snr_db)
     962 +        self._snr_hist.append(float(snr_db))
     963 +
     964 +        prev_active = self._vad_active
     965 +        speech_now = self._vad_is_speech(rms, snr_db)
     966 +        if speech_now:
     967 +            self._vad_active = True
     968 +            self._vad_hangover = int(self._vad_hangover_frames)
     969 +        else:
     970 +            if self._vad_active:
     971 +                self._vad_hangover -= 1
     972 +                if self._vad_hangover <= 0:
     973 +                    self._vad_active = False
     974 +            else:
     975 +                self._vad_active = False
     976 +
     977 +        if not self._vad_active:
     978 +            self._update_noise_floor(rms)
     979 +            if self._nr_enabled:
     980 +                self._update_noise_profile(chunk)
     981              # End utterance buffer if we have been collecting.
         ⋮
     983                  self._finalize_utterance_harvest()
     984 +            if self._preroll.maxlen and self._preroll.maxlen > 0:
     985 +                self._preroll.append(chunk)
     986              return
         ⋮
    1017
     867 -        x = chunk.astype(np.float32) * float(sensitivity)
    1018 +        if not prev_active and self._preroll:
    1019 +            for pr in self._preroll:
    1020 +                xr = pr
    1021 +                if self._nr_enabled:
    1022 +                    xr = self._denoise_chunk(xr)
    1023 +                xr = xr.astype(np.float32, copy=False) * float(sensiti
          vity)
    1024 +                self._asr_stream.accept_waveform(self._sample_rate, xr
          )
    1025 +            self._preroll.clear()
    1026 +
    1027 +        x = chunk
    1028 +        if self._nr_enabled:
    1029 +            x = self._denoise_chunk(x)
    1030 +        x = x.astype(np.float32, copy=False) * float(sensitivity)
    1031          self._asr_stream.accept_waveform(self._sample_rate, x)

• Ran rg -n "def _speak|return$|return\s+#" -n backend/goeckoh/systems/
  │ speech_mirror.py
  └ 782:                return
    846:            return
    … +25 lines
    1507:            return
    1526:            return

