from google.adk.agents import Agent, SequentialAgent
from pydantic import BaseModel, Field

class SpeechOutput(BaseModel):
    corrected_text: str = Field(description="The grammatically corrected first-person speech.")

