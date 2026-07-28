from pydantic import BaseModel
from backend.app.config import settings
from backend.app.guardrails.exceptions import InputValidationException, SQLGuardrailException
import re

class ValidationResult(BaseModel):
    is_valid: bool
    reason: str = ""

def validate_question(question: str) -> ValidationResult:
    """Validates the input question against basic sanity rules."""
    if not settings.ENABLE_GUARDRAILS:
        return ValidationResult(is_valid=True)
        
    if not question or not question.strip():
        raise InputValidationException("Question cannot be empty or only whitespace.")
        
    if len(question) > settings.MAX_QUESTION_LENGTH:
        raise InputValidationException(f"Question exceeds maximum length of {settings.MAX_QUESTION_LENGTH} characters.")
        
    # Check for control characters or binary (basic heuristic)
    if any(ord(c) < 32 and c not in '\n\r\t' for c in question):
        raise InputValidationException("Question contains invalid control characters.")
        
    # Detect extremely repeated tokens (e.g. "a a a a a a...")
    tokens = question.split()
    if len(tokens) > 20 and len(set(tokens)) == 1:
        raise InputValidationException("Question contains extremely repeated tokens.")
        
    return ValidationResult(is_valid=True)

def validate_sql(sql: str) -> ValidationResult:
    """Validates SQL before execution to prevent destructive commands."""
    if not settings.ENABLE_SQL_GUARD or not settings.ENABLE_GUARDRAILS:
        return ValidationResult(is_valid=True)
        
    sql_upper = sql.upper()
    
    # Only allow SELECT
    if not sql_upper.strip().startswith("SELECT"):
        raise SQLGuardrailException("Only SELECT statements are allowed.")
        
    forbidden_keywords = [
        "DELETE", "INSERT", "DROP", "ALTER", "UPDATE", "CREATE", 
        "TRUNCATE", "ATTACH", "PRAGMA", "VACUUM"
    ]
    
    for kw in forbidden_keywords:
        if re.search(rf"\b{kw}\b", sql_upper):
            raise SQLGuardrailException(f"SQL contains forbidden keyword: {kw}")
            
    # Reject multiple statements
    if ";" in sql:
        raise SQLGuardrailException("Multiple SQL statements are not allowed.")
        
    # Reject comments
    if "--" in sql or "/*" in sql or "*/" in sql:
        raise SQLGuardrailException("SQL comments are not allowed.")
        
    # Reject UNION based attacks
    if "UNION" in sql_upper:
        raise SQLGuardrailException("UNION statements are not allowed.")
        
    return ValidationResult(is_valid=True)
