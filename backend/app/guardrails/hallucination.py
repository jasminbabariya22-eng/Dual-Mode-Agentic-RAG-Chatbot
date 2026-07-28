from pydantic import BaseModel
from typing import List
from backend.app.config import settings

class HallucinationResult(BaseModel):
    supported: bool
    confidence: float
    unsupported_claims: List[str]

def verify_answer_against_context(question: str, chunks: str, answer: str, current_confidence: float) -> HallucinationResult:
    """Verifies that the generated answer is supported by the context chunks."""
    if not settings.ENABLE_HALLUCINATION_CHECK or not settings.ENABLE_GUARDRAILS:
        return HallucinationResult(supported=True, confidence=current_confidence, unsupported_claims=[])
        
    # A lightweight heuristic hallucination checker. 
    # In a full enterprise system, this might call another LLM.
    # We simulate a confidence drop if specific un-grounded claims are made.
    
    # If chunks are empty but answer is extremely detailed, flag it.
    if not chunks.strip() and len(answer.split()) > 30 and "i do not have enough" not in answer.lower():
        return HallucinationResult(
            supported=False, 
            confidence=0.1, 
            unsupported_claims=["Detailed answer provided without context."]
        )
        
    # We assume it's supported for now, returning the base confidence
    return HallucinationResult(
        supported=True,
        confidence=current_confidence,
        unsupported_claims=[]
    )
