from google.adk.agents import Agent, SequentialAgent
from pydantic import BaseModel, Field

class SpeechSchema(BaseModel):
    text: str = Field(description="The grammatically corrected first-person text.")

