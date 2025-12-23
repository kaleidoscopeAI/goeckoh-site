// - Integrated RNNoise via WebRTC noise suppression using JNI wrapper (based on WebRTC's Ns module, which uses RNNoise tech). Assumed libwebrtc_ns_jni.so built and loaded via System.loadLibrary.
// - Added real-time VAD metrics UI: TextView in layout, updated via LocalBroadcastManager from service on utterance end (speech rate, avg latency, total frames, speech frames).
// - NsWrapper class with native methods for noise suppression, applied to short[] buffers before VAD/Whisper.
