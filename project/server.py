from quart import Quart, jsonify, request
from quart_cors import cors as cors_decorator
from datetime import datetime
from pathlib import Path
import asyncio

from backend.core.router import (
    process_sensory_packet,
    pull_echo_commands,
    get_latest_state,
)
from backend.guardian.policy import GuardianPolicy
from backend.guardian.gate import action_allowed, queue_for_guardian_review # Import queue_for_guardian_review
from backend.engines.local_llm_core import LocalLLMCore, LocalLLMConfig # NEW: Import LLM core

app = Quart(__name__)
app = cors_decorator(app, allow_origin="*")

GUARDIAN_POLICY_PATH = Path("./config/guardian_policy.json") # Central policy config
guardian_policy = GuardianPolicy.load(GUARDIAN_POLICY_PATH)

# Store pending suggestions for the Guardian HUD
pending_guardian_suggestions = []

def init_llm() -> None:
    """Initializes the global LocalLLMCore instance."""
    # Resolve path relative to this server.py file
    model_path = Path(__file__).resolve().parent.parent.parent / "models" / "llm" / "echo_brain.Q4_K_M.gguf"
    cfg = LocalLLMConfig(
        model_path=model_path,
        n_ctx=4096,
        n_threads=8, # Adjust based on CPU cores
        n_gpu_layers=0,  # Set > 0 if you have a GPU and llama.cpp is built with GPU support
    )
    LocalLLMCore.init_global(cfg)
    app.logger.info(f"LocalLLMCore initialized with model: {model_path}")

@app.before_serving
async def startup() -> None:
    # Initialize the local LLM once when the backend starts
    init_llm()
    app.logger.info("CCA Backend started and LLM initialized.")

@app.post("/api/sensory/packet")
async def sensory_packet():
    data = await request.get_json()
    process_sensory_packet(data) # This will now use thought_pipeline, which uses ABAPolicyEngine and LLMReasoner
    return jsonify({"status": "ok"}), 200

@app.get("/api/commands/pull")
async def commands_pull():
    session_id = request.args.get("session_id", "") or "default"
    since = float(request.args.get("since", "0.0"))
    cmds = pull_echo_commands(session_id, since)
    return jsonify(cmds), 200

# NEW HID PULL (placeholder for now)
@app.get("/api/commands/hid_pull")
async def commands_hid_pull():
    session_id = request.args.get("session_id", "") or "default"
    since = float(request.args.get("since", "0.0"))
    # In a real implementation, pull HID commands from a buffer similar to echo_commands
    return jsonify([]), 200 # No HID commands for now

@app.get("/api/state/echo")
async def state_echo():
    session_id = request.args.get("session_id", "") or "default"
    state = get_latest_state(session_id)
    return jsonify(state), 200

@app.get("/api/guardian/policy")
async def get_guardian_policy():
    return jsonify(guardian_policy.raw), 200

@app.put("/api/guardian/policy")
async def put_guardian_policy():
    global guardian_policy
    try:
        new_policy_data = await request.get_json()
        guardian_policy = GuardianPolicy(raw=new_policy_data)
        guardian_policy.save(GUARDIAN_POLICY_PATH)
        return jsonify({"status": "ok", "message": "Guardian policy updated and saved."}), 200
    except Exception as e:
        app.logger.error(f"Error updating guardian policy: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.get("/api/guardian/suggestions")
async def get_guardian_suggestions():
    # In a real implementation, you'd integrate the queue_for_guardian_review
    # to actually store suggestions here. For now, it's a simple list.
    # We need a thread-safe way to manage this.
    suggestions_copy = pending_guardian_suggestions[:] # Get a copy
    pending_guardian_suggestions.clear() # Clear after serving
    return jsonify(suggestions_copy), 200

def main():
    # Ensure the config directory exists for policy storage
    GUARDIAN_POLICY_PATH.parent.mkdir(parents=True, exist_ok=True)
    app.run(host="127.0.0.1", port=5000)

if __name__ == "__main__":
    main()