from backend.app.config import settings

print("LLM_PROVIDER =", settings.LLM_PROVIDER)
print("LLM_MODEL =", settings.LLM_MODEL)
print("OLLAMA_BASE_URL =", settings.OLLAMA_BASE_URL)
print("ENABLE_FALLBACK_MODEL =", settings.ENABLE_FALLBACK_MODEL)
print("GROQ_API_KEY =", settings.GROQ_API_KEY)