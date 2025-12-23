// - Noise suppression: Integrated using Android's NoiseSuppressor (AudioEffect), applied to audio chunks before VAD and Whisper. Check availability, fallback if not supported.
// - VAD performance metrics: Track and log metrics - processing time per frame, speech detection rate (speech frames / total), false positives (manual sim, but log probs), latency (time from audio to detection).
