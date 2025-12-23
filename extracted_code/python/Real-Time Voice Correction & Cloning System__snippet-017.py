from google.adk.agents import Agent, SequentialAgent
from pydantic import BaseModel, Field

class SpeechCorrection(BaseModel):
    text: str = Field(description="Corrected first-person text.")

