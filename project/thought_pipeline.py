"""
Thought pipeline that converts a sensory packet + memory context into a list of
actions Echo can execute (speech + ABA strategies).
"""

from __future__ import annotations

from typing import Any, Dict, List
from backend.guardian.policy import GuardianPolicy # Added import

from backend.engines.aba_policy import ABAPolicyEngine
from backend.engines.llm_reasoning import LLMReasoner


class ThoughtPipeline:
    def __init__(self) -> None:
        self.aba = ABAPolicyEngine()
        self.llm = LLMReasoner()

    def decide(self, packet: Dict[str, Any], memory, policy: GuardianPolicy) -> List[Dict[str, Any]]: # Added policy argument
        """
        Takes a single sensory packet and global memory and returns a list of
        action dicts to be sent back to Echo.
        """
        raw_text = packet.get("raw_text", "")
        clean_text = packet.get("clean_text", "")
        emotion = packet.get("emotion") or {}
        aba_ctx = packet.get("aba") or {}

        history = memory.recent_context(6)
        aba_plan = self.aba.select_plan(aba_ctx, history, emotion, policy) # Pass policy to ABA

        llm_out = self.llm.generate_supportive_response(
            raw_text=raw_text,
            clean_text=clean_text,
            emotion=emotion,
            aba_plan=aba_plan,
            history=history,
            policy=policy, # Pass policy to LLM
        )

        mirror_text = llm_out.get("mirror_text") or clean_text
        prosody = llm_out.get("prosody") or {
            "rate": 0.95,
            "pitch_shift": 0.0,
            "energy": 0.8,
            "style": "calm_supportive",
        }

        actions: List[Dict[str, Any]] = [
            {
                "type": "speech",
                "channel": "child_voice",
                "mode": "mirror_correct",
                "text": mirror_text,
                "prosody": prosody,
            }
        ]

        strategy_id = aba_plan.get("strategy_id")
        if strategy_id:
            actions.append(
                {
                    "type": "aba",
                    "strategy_id": strategy_id,
                    "parameters": aba_plan.get("parameters", {}),
                }
            )

        return actions
