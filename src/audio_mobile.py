from jnius import autoclass
from threading import Thread
import struct
import numpy as np

AudioRecord = autoclass('android.media.AudioRecord')
AudioTrack = autoclass('android.media.AudioTrack')

class AndroidAudioBridge:
    def __init__(self, mic_q, spk_q, sr: int):
        self.mic_q = mic_q
        self.spk_q = spk_q
        self.sr = sr
        self.running = False
        self.buf_size = AudioRecord.getMinBufferSize(sr, 16, 2)

    def start(self):
        self.running = True
        Thread(target=self._run, daemon=True).start()

    def _run(self):
        rec = AudioRecord(6, self.sr, 16, 2, self.buf_size * 2)
        trk = AudioTrack(3, self.sr, 4, 2, self.buf_size * 2, 1)
        rec.startRecording()
        trk.play()
        
        buf = bytearray(self.buf_size)
        while self.running:
            n = rec.read(buf, 0, len(buf))
            if n > 0:
                ints = struct.unpack(f"<{n//2}h", buf[:n])
                floats = np.array(ints, dtype=np.float32) / 32768.0
                self.mic_q.put(floats.reshape(-1, 1))
            
            if not self.spk_q.empty():
                data = self.spk_q.get()
                shorts = (np.clip(data, -1, 1) * 32767).astype(np.int16)
                trk.write(shorts.tobytes(), 0, len(shorts) * 2)
