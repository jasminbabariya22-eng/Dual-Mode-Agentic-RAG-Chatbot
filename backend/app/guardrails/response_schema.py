from typing import List
from pydantic import BaseModel, Field

class GuardedAnswer(BaseModel):
    """Structured response definition for output validation."""
    answer: str = Field(description="The final answered text")
    citations: List[str] = Field(description="List of document citations")
    confidence: float = Field(description="Confidence score for hallucination/router")
    route: str = Field(description="The execution route taken")
    validated: bool = Field(description="Whether output passed guardrails")
    hallucination_score: float = Field(description="Score representing factual groundedness")
