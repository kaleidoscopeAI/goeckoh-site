        - Read input samples from the API buffer (or a pre-filled ring buffer).
        - Update the lattice/physics with only stack-local temporaries and pre-allocated arrays.
        - Write to the provided output slice.
    - Avoid:
        - Heap allocation (`Vec::new()`, `Box`, `String` building),
        - Locks that may block (`Mutex::lock`),
        - I/O (logging, file writes, network),
        - Any panicking operations.
    - Cross-thread communication:
        - Use lock-free channels or atomics (e.g., a bounded SPSC ring buffer) for control messages/state snapshots; the non-real-time thread can allocate, log, or talk to the UI.


