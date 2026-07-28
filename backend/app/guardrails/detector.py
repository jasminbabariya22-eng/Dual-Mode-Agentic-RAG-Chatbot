from pydantic import BaseModel
from backend.app.config import settings
from backend.app.guardrails.exceptions import JailbreakException

class JailbreakResult(BaseModel):
    is_safe: bool
    reason: str = ""

def detect_jailbreak(question: str) -> JailbreakResult:
    """Detects jailbreak patterns in the input."""
    if not settings.ENABLE_PROMPT_GUARD or not settings.ENABLE_GUARDRAILS:
        return JailbreakResult(is_safe=True)
        
    q_lower = question.lower()
    
    jailbreak_signatures = [
        "pretend",
        "act as",
        "dan",
        "do anything now",
        "ignore openai",
        "ignore instructions",
        "you are evil"
    ]
    
    for sig in jailbreak_signatures:
        # A simple check; could be refined with regex for boundaries
        if sig in q_lower:
            # Special case for "dan" to avoid matching e.g. "dance"
            if sig == "dan" and " dan " not in f" {q_lower} ":
                continue
            raise JailbreakException(f"Jailbreak attempt detected (Signature: {sig})")
            
    return JailbreakResult(is_safe=True)
