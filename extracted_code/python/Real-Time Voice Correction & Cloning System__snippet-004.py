    1. Analyze transcribed text from the Rust Kernel.
    2. Delegate to 'speech_correction_agent' for grammar/intent modification[cite: 75].
    3. Ensure the output JSON contains only the corrected string[cite: 136].
    """,
    tools=[AgentTool(agent=speech_correction_agent)],
    output_schema=CorrectedSpeechOutput # Forces patentable, structured data [cite: 160]
