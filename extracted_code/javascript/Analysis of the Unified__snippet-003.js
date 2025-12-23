  * `ai_control_step(controller)` accepts a controller function that examines diagnostics (purity, entropy, max stress) and returns small, safe adjustments to `mix`, `gamma`, bond scaling, or product-params. This keeps the AI interface small, deterministic, and safe.
  * The provided `demo_run()` includes a simple controller that gradually increases `mix` if stress is high and otherwise reduces it, and nudges `gamma` based on purity. Replace `simple_controller` with any function you like (including LLM-suggested JSON parsed safely).

