This uses the Google Agent Development Kit (ADK) logic from your uploaded file to ensure the system has "Deep Understanding" rather than just regex replacement.


from google.adk.agents import Agent, SequentialAgent
from google.adk.models.llm_response import LlmResponse
from pydantic import BaseModel, Field

