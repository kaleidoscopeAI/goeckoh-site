name="speech_correction_agent",
description="Corrects grammar and pronouns for real-time streaming.",
instruction="""You are an expert linguist. 
1. Summarize the user's intent from the {transcription} key in session state[cite: 159].
2. Apply groundbreaking corrections: change third-person to first-person[cite: 31, 44].
3. Return ONLY valid JSON matching the schema[cite: 136, 160].""",
output_schema=SpeechCorrectionOutput
