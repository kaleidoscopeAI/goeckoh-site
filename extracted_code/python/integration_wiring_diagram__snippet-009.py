    *   **Data In:** Real-time emotional intensity, ABA strategy logs, Organic AI node count/energy from `backend/server.py`.
    *   **Commands Out:** Manual intervention overrides from `echo_gui.py` to `backend/server.py`.
    *   **Flow:**
        `echo_gui.py` <---WebSocket/REST---> `backend/server.py` <---Internal API/IPC---> `agireal/onbrain_autonomous.py` / `echo_prime.py`.

    This option is generally more feasible, treating `echo_gui.py` as a specialized client.

