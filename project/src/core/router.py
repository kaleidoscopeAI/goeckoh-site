"""
Core routing between Echo sensory packets, crystalline memory, thought engines,
and command buffers for Echo + HID.

This is the minimal integration glue between echovoice/echo_prime.py and the
Cognitive Crystal AI backend.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional
from pathlib import Path # Required for GuardianPolicy.load

from backend.memory.crystalline_memory import CrystallineMemory
from backend.engines.thought_pipeline import ThoughtPipeline
from backend.guardian.policy import GuardianPolicy
from backend.server import GUARDIAN_POLICY_PATH # Import the path from server

# Global singletons
_memory = CrystallineMemory()
_pipeline = ThoughtPipeline()
_guardian_policy = GuardianPolicy.load(GUARDIAN_POLICY_PATH) # Load policy globally

# session_id -> list of command dicts
_command_buffer: Dict[str, List[Dict[str, Any]]] = {}
_buffer_lock = threading.Lock()


def process_sensory_packet(packet: Dict[str, Any]) -> None:
    """
    Entry point called by backend.server when /api/sensory/packet is hit.

    - Stores packet in crystalline memory
    - Runs thought pipeline to derive actions
    - Appends actions to the per-session command buffer
    """
    session_id = packet.get("session_id") or "default"
    now = time.time()

    # 1. Store into memory
    _memory.store_packet(packet)

    # 2. Derive decisions (actions) - Pass the guardian policy here
    decisions = _pipeline.decide(packet, _memory, _guardian_policy)

    if not decisions:
        return

    cmd = {
        "session_id": session_id,
        "utterance_id": (packet.get("echo_meta") or {}).get("utterance_id", ""),
        "ts": now,
        "actions": decisions,
    }

    # 3. Append to buffer
    with _buffer_lock:
        buf = _command_buffer.setdefault(session_id, [])
        buf.append(cmd)


def pull_echo_commands(session_id: str, since_ts: float) -> List[Dict[str, Any]]:
    """
    Returns all commands for the given session whose timestamp > since_ts.
    Intended to be called by /api/commands/pull.
    """
    if not session_id:
        session_id = "default"

    with _buffer_lock:
        buf = _command_buffer.get(session_id, [])
        return [c for c in buf if float(c.get("ts", 0.0)) > since_ts]


def pull_hid_commands(session_id: str, since_ts: float) -> List[Dict[str, Any]]:
    """
    Returns all HID commands for the given session whose timestamp > since_ts.
    Intended to be called by /api/commands/hid_pull.
    For now, this is a placeholder.
    """
    if not session_id:
        session_id = "default"
    
    # In a full implementation, you'd have a separate HID command buffer
    # For now, we'll just return an empty list
    return []

def get_latest_state(session_id: str) -> Dict[str, Any]:
    """
    Minimal state view for GUIs (echo_gui, web frontend).
    """
    if not session_id:
        session_id = "default"

    recent = _memory.recent_context(1)
    last: Optional[Dict[str, Any]] = recent[-1] if recent else None

    with _buffer_lock:
        buf = _command_buffer.get(session_id, [])
        last_cmd: Optional[Dict[str, Any]] = buf[-1] if buf else None

    state: Dict[str, Any] = {
        "session_id": session_id,
        "turn_index": None,
        "raw_text": None,
        "clean_text": None,
        "emotion": None,
        "last_actions": last_cmd["actions"] if last_cmd else [],
    }

    if last:
        state["turn_index"] = (last.get("echo_meta") or {}).get("turn_index")
        state["raw_text"] = last.get("raw_text")
        state["clean_text"] = last.get("clean_text")
        state["emotion"] = last.get("emotion")

    return state