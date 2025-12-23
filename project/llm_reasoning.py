"""
LLMReasoner: high-level therapeutic planning around Echo.

Uses LocalLLMCore (llama.cpp) to choose:
- mirror_text (first-person correction to be spoken in child's voice)
- prosody (rate, pitch_shift, energy, style)
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from backend.engines.local_llm_core import LocalLLMCore
from backend.guardian.policy import GuardianPolicy


class LLMReasoner:
    def __init__(self) -> None:
        # No config here; LocalLLMCore is initialized once at startup.
        self.llm = LocalLLMCore.get_global()

    def generate_supportive_response(
        self,
        raw_text: str,
        clean_text: str,
        emotion: Dict[str, Any],
        aba_plan: Dict[str, Any],
        history: List[Dict[str, Any]],
        policy: GuardianPolicy,
    ) -> Dict[str, Any]:
        """
        Returns:
            {
              "mirror_text": "...",
              "prosody": { "rate": ..., "pitch_shift": ..., "energy": ..., "style": "..." }
            }
        """

        # Only use a short, anonymized slice of history in the prompt
        short_hist = [
            {
                "raw_text": h.get("raw_text", ""),
                "clean_text": h.get("clean_text", ""),
                "emotion": h.get("emotion", {}),
                "aba": h.get("aba", {}),
            }
            for h in history[-4:]
        ]

        payload = {
            "raw_text": raw_text,
            "clean_text": clean_text,
            "emotion": emotion,
            "aba_plan": aba_plan,
            "recent_history": short_hist,
        }

        # Add policy-driven instructions to the system prompt
        system_policy_context = ""
        if policy.raw.get("echo", {}).get("corrections_enabled"):
            system_policy_context += "Prioritize grammar and clarity in mirror_text. "
        else:
            system_policy_context += "Preserve the child's exact phrasing if grammatically understandable, only correcting major errors. "
        
        if policy.raw.get("echo", {}).get("night_mode_quiet_hours", {}).get("enabled"):
            system_policy_context += "It is night mode, keep responses extra calm and quiet, suggest lower volume. "


        system_prompt = (
            "You are the cognitive planning layer for an autism speech companion.\n"
            "You DO NOT speak directly to the child. You decide what the companion "
            "will say in the child's own voice.\n\n"
            "Rules:\n"
            "1. You always speak in FIRST PERSON (I / me / my).\n"
            "2. You preserve the child's intent but fix grammar and clarity.\n"
            "3. You keep responses short and concrete.\n"
            "4. You output STRICT JSON with keys 'mirror_text' and 'prosody'.\n"
            f"{system_policy_context}"
        )

        user_prompt = (
            "Based on the following data, produce JSON:\n\n"
            f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
            "Output format (no explanation, no extra keys):\n"
            "{\n"
            '  "mirror_text": "...",\n'
            '  "prosody": { "rate": float, "pitch_shift": float, ' 
            '"energy": float, "style": "string" }\n'
            "}"
        )

        raw = self.llm.chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            stop=["</s>", "<<USER>>", "<<SYS>>"],
            temperature=0.4,
            max_tokens=256,
        )

        try:
            data = json.loads(raw)
        except Exception:
            # Safe fallback if model output isn't valid JSON
            data = {
                "mirror_text": clean_text,
                "prosody": {
                    "rate": 0.95,
                    "pitch_shift": 0.0,
                    "energy": 0.8,
                    "style": "calm_supportive",
                },
            }
        return data
