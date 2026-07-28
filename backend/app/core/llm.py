"""
LLM Factory

Provides a single entry point for creating LangChain chat models.

Primary Provider:
    - Ollama (Local)

Fallback Provider:
    - Groq (Cloud)

Designed for LangGraph and FastAPI applications.
"""

from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel

from backend.app.config import settings
from backend.app.core.logger import logger


def _create_llm_instance(
    provider: str,
    model: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> BaseChatModel:
    """
    Create a LangChain ChatModel for the given provider.
    """

    provider = provider.lower()

    # ==========================================================
    # Ollama
    # ==========================================================
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model,
            base_url=base_url or settings.OLLAMA_BASE_URL,
            temperature=settings.LLM_TEMPERATURE,
        )

    # ==========================================================
    # Groq
    # ==========================================================
    elif provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=model,
            api_key=api_key or settings.GROQ_API_KEY,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
            timeout=settings.LLM_TIMEOUT,
            max_retries=settings.LLM_MAX_RETRIES,
        )

    # ==========================================================
    # OpenAI (Optional)
    # ==========================================================
    elif provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model,
            api_key=api_key,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
            timeout=settings.LLM_TIMEOUT,
            max_retries=settings.LLM_MAX_RETRIES,
        )

    # ==========================================================
    # Gemini (Optional)
    # ==========================================================
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=settings.LLM_TEMPERATURE,
            max_output_tokens=settings.LLM_MAX_TOKENS,
        )

    else:
        raise ValueError(f"Unsupported LLM Provider: {provider}")


def get_llm(use_fallback: bool = True) -> BaseChatModel:
    """
    Returns the configured LLM.

    Primary:
        Ollama

    Fallback:
        Groq

    If fallback is enabled, LangChain automatically switches
    to the fallback model when the primary model fails.
    """

    logger.info(
        f"Initializing Primary LLM: "
        f"{settings.LLM_PROVIDER}:{settings.LLM_MODEL}"
    )

    primary = _create_llm_instance(
        provider=settings.LLM_PROVIDER,
        model=settings.LLM_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
    )

    if not use_fallback or not settings.ENABLE_FALLBACK_MODEL:
        return primary

    # Guard: only build the fallback chain when the API key is actually present.
    # An empty string passes ChatGroq.__init__ but raises AuthenticationError at
    # invocation time, which poisons the entire RunnableWithFallbacks chain and
    # produces a silent FallbackException instead of a real answer.
    if not settings.GROQ_API_KEY:
        logger.warning(
            "Groq API key is absent — fallback LLM disabled. "
            "Primary LLM (Ollama) will be used without a fallback chain."
        )
        return primary

    try:
        logger.info(
            f"Initializing Fallback LLM: "
            f"{settings.FALLBACK_PROVIDER}:{settings.FALLBACK_MODEL}"
        )

        fallback = _create_llm_instance(
            provider=settings.FALLBACK_PROVIDER,
            model=settings.FALLBACK_MODEL,
            api_key=settings.GROQ_API_KEY,
        )

        logger.info("Fallback chain enabled.")

        return primary.with_fallbacks([fallback])

    except Exception as ex:
        logger.warning(
            f"Failed to initialize fallback model: {ex}. "
            "Primary LLM will be used without a fallback chain."
        )
        return primary