This defines the semantic correction layer using Google ADK design patterns for deterministic pipelines.


from google.adk.agents import Agent, SequentialAgent
from google.adk.models.llm_response import LlmResponse
from pydantic import BaseModel, Field

class SpeechOutput(BaseModel):
    corrected_text: str = Field(description="The finalized first-person speech string.")

