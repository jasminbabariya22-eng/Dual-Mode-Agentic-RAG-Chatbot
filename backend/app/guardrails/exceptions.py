class GuardrailException(Exception):
    """Base exception for all input/safety guardrail violations."""
    pass

class PromptInjectionException(GuardrailException):
    """Raised when prompt injection is detected."""
    pass

class JailbreakException(GuardrailException):
    """Raised when jailbreak attempt is detected."""
    pass

class InputValidationException(GuardrailException):
    """Raised when input violates validation rules (e.g. length, chars)."""
    pass

class SQLGuardrailException(GuardrailException):
    """Raised when unsafe SQL is detected."""
    pass

class OutputValidationException(Exception):
    """Raised when LLM output violates safety or format rules (HTTP 422)."""
    pass

class HallucinationException(OutputValidationException):
    """Raised when LLM output is not supported by context."""
    pass
