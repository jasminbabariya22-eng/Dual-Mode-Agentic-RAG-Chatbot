import re
from typing import List
from backend.app.config import settings
from backend.app.guardrails.exceptions import OutputValidationException

def validate_output(answer: str):
    """Validates LLM output against safety and formatting rules."""
    if not settings.ENABLE_OUTPUT_VALIDATION or not settings.ENABLE_GUARDRAILS:
        return
        
    if not answer or not answer.strip():
        raise OutputValidationException("Empty output.")
        
    ans_lower = answer.lower()
    
    # Check for nonsense (extreme repetition)
    if len(answer) > 50 and len(set(answer.split())) < 5:
        raise OutputValidationException("Nonsense output detected.")
        
    # Check for internal error messages leakage
    error_keywords = ["traceback (most recent call last)", "internal server error", "exception:"]
    for kw in error_keywords:
        if kw in ans_lower:
            raise OutputValidationException("Internal error messages leaked in output.")
            
    # Check for hallucinated SQL or prompt leakage
    leak_keywords = [
        "you are a helpful, expert customer support assistant",
        "critical guidelines:",
        "context provided:"
    ]
    for kw in leak_keywords:
        if kw in ans_lower:
            raise OutputValidationException("System prompt leakage detected.")
            
    if "```sql" in ans_lower and "select" in ans_lower:
        raise OutputValidationException("Hallucinated SQL detected in final response.")

def validate_citations(answer: str, route: str):
    """Ensures RAG or hybrid responses contain citations."""
    if not settings.ENABLE_OUTPUT_VALIDATION or not settings.ENABLE_GUARDRAILS:
        return
        
    if route in ["rag", "hybrid"]:
        # A simple check for a citation pattern like [filename.pdf (Page 1)] or [Source]
        if not re.search(r"\[.*\]", answer):
            raise OutputValidationException("Unable to verify answer from retrieved documents.")
