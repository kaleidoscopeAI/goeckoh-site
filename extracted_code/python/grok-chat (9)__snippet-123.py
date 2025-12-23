  - Rust voice path is built and optional (GOECKOH_USE_RUST_VC=1); playback
    and processing latencies are tracked, with p95 targets enforced via
    validation endpoints.
  - Mirror metrics and validation endpoints expose heart coherence/
    activation/valence (from the Rust heart engine), latency p95/playback
    p95, drift p95, and gate blocks.
  - Dashboard shows mirror heart state and validation targets; validation/
    loopback scripts are ready but need real session data to populate logs.

