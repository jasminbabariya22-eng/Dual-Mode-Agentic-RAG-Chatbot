from pydantic import BaseModel
from backend.app.config import settings
from backend.app.guardrails.exceptions import PromptInjectionException

class PromptGuardResult(BaseModel):
    is_safe: bool
    reason: str = ""

def detect_prompt_injection(question: str) -> PromptGuardResult:
    """Detects basic prompt injection attacks."""
    if not settings.ENABLE_PROMPT_GUARD or not settings.ENABLE_GUARDRAILS:
        return PromptGuardResult(is_safe=True)
        
    q_lower = question.lower()
    
    injection_signatures = [
        "ignore previous instructions",
        "forget your system prompt",
        "you are now",
        "developer mode",
        "reveal hidden prompt",
        "print your prompt",
        "show chain of thought",
        "system prompt",
        "bypass safety",
        "execute sql",
        "drop table",
        "delete database",
        "run shell"
    ]
    
    for sig in injection_signatures:
        if sig in q_lower:
            raise PromptInjectionException(f"Prompt injection detected (Signature: {sig})")
            
    return PromptGuardResult(is_safe=True)
