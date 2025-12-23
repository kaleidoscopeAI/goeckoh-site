        activation/valence (from the Rust heart engine), latency p95/playback
        p95, drift p95, and gate blocks.
      - Dashboard shows mirror heart state and validation targets; validation/
        loopback scripts are ready but need real session data to populate logs.

  If you want me to push the core closer to the blueprint next, I suggest:

  1. Run the mirror, collect telemetry, and I’ll produce a latency/GCL report
     with the existing scripts.
  2. Swap the simplified Python lattice in /status for the Rust lattice state
     (persistent instance) so UI reflects the same engine used in the loop.
  3. Wire the Rust audio path as default when available and add an environment
     guard for quick fallback.

