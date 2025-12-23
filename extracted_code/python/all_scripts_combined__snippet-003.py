class AGISeed:
    """The central decision-making gear of the digital organism."""

    def decide(self, emotion_info: Information, speech_info: Optional[Information]) -> AgentDecision:
        """
        Given the current emotional and speech state, decide on an action.
        """
        if not isinstance(emotion_info.payload, EmotionData):
            return AgentDecision(action="IDLE")

        emotion: EmotionData = emotion_info.payload
        speech: Optional[SpeechData] = speech_info.payload if speech_info else None

        # 1. High-priority Regulation
        # If arousal is very high and coherence is low, the system is "distressed."
        # The top priority is to calm down (regulate).
        if emotion.arousal > 7.0 and emotion.coherence < 0.3:
            return AgentDecision(
                action="REGULATE",
                target_text="I am breathing. In... and out.",
                mode="inner"
            )

        # 2. Echoing User Speech
        # If there's speech to process, the default action is to echo it.
        if speech and (speech.corrected_text or speech.raw_text):
            # If the user's speech was successfully corrected, the system is "learning"
            # by reinforcing the correct pattern.
            if speech.corrected_text and speech.raw_text != speech.corrected_text:
                return AgentDecision(
                    action="LEARN",
                    target_text=speech.corrected_text,
                    mode="inner" # Corrections are always inner voice
                )
            # Otherwise, just a simple echo.
            return AgentDecision(
                action="ECHO",
                target_text=speech.raw_text,
                mode="inner"
            )
            
        # 3. Expressing an Internal Thought from the LLM
        # If the LLM generated a thought and the system is not in a high-arousal state.
        if emotion_info.metadata.get("llm_output") and emotion.arousal < 5.0:
            return AgentDecision(
                action="THINK",
                target_text=emotion_info.metadata["llm_output"],
                mode="inner" # Internal thoughts are spoken with the inner voice
            )

        # 4. Default Idle State
        # If none of the above conditions are met, do nothing.
        return AgentDecision(action="IDLE")

