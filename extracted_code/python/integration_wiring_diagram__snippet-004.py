    *   **Receives:** Text, Emotional Intensity, (optional) Raw Audio from `echo_prime.py`.
    *   **Mechanism:** `echo_prime.py` would send this data via HTTP POST requests to specific endpoints exposed by `backend/server.py`.
    *   **Routes:**
        *   `/api/sensory/audio`: For raw audio/features.
        *   `/api/sensory/text`: For transcribed text.
        *   `/api/sensory/emotion`: For arousal/valence updates.

