import torch
import numpy as np
import sounddevice as sd
from typing import Generator, Optional
from .config import AudioSettings
from .gears import Information, AudioData

class AudioInputGear:
    """
    Listens to the microphone and uses Silero VAD to yield speech chunks.
    """
    def __init__(self, audio_cfg: AudioSettings, device: str = "cpu"):
        self.config = audio_cfg
        self.device = device
        
        # Load Silero VAD model
        try:
            self.model, self.utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False # Set to True for first run if needed
            )
            (self.get_speech_timestamps, _, self.read_audio, _, _) = self.utils
        except Exception as e:
            raise RuntimeError(f"Failed to load Silero VAD model. Have you run it once with force_reload=True? Error: {e}")

        self.stream = sd.InputStream(
            samplerate=self.config.sample_rate,
            channels=self.config.channels,
            dtype='float32'
        )

    def listen(self) -> Generator[Information, None, None]:
        """
        A generator that continuously listens to the microphone and yields
        Information objects containing speech audio data.
        """
        print("Sensory Gear Activated. Listening...")
        self.stream.start()
        
        # VAD requires a buffer to work on
        buffer = np.array([], dtype=np.float32)
        
        while True:
            # Read from the microphone stream
            chunk, overflowed = self.stream.read(self.stream.read_available)
            if overflowed:
                print("Warning: audio input overflowed")
            
            if chunk.size > 0:
                buffer = np.concatenate([buffer, chunk[:, 0]]) # Mono

            # Process buffer if it's large enough
            # VAD works best on chunks of a few seconds
            if len(buffer) > self.config.sample_rate * 2:
                try:
                    # Use VAD to find speech timestamps in the buffer
                    speech_timestamps = self.get_speech_timestamps(
                        torch.from_numpy(buffer),
                        self.model,
                        threshold=self.config.vad_threshold,
                        min_silence_duration_ms=self.config.vad_min_silence_duration_ms,
                        speech_pad_ms=self.config.vad_speech_pad_ms,
                    )

                    if speech_timestamps:
                        # Extract the first detected speech chunk
                        start = speech_timestamps[0]['start']
                        end = speech_timestamps[0]['end']
                        speech_chunk = buffer[start:end]
                        
                        # Yield the speech chunk as an Information object
                        yield Information(
                            payload=AudioData(waveform=speech_chunk, sample_rate=self.config.sample_rate),
                            source_gear="AudioInputGear"
                        )
                        
                        # The remaining part of the buffer is what's left after the speech
                        buffer = buffer[end:]
                    else:
                        # No speech detected, keep the last second of audio in buffer
                        # to handle speech that might cross buffer boundaries.
                        buffer = buffer[-self.config.sample_rate:]

                except Exception as e:
                    print(f"Error during VAD processing: {e}")
                    # Reset buffer on error
                    buffer = np.array([], dtype=np.float32)-e 


