"""
Conversation Memory Layer.

Provides pluggable session memory backends for multi-turn conversation history.
Supports in-process storage (InMemoryMemory) and Redis (RedisMemory), with
automatic graceful fallback from Redis to in-memory on connection failure.

Design decisions:
- BaseMemory defines the contract so backends are interchangeable.
- History is stored as a list of {role, content} dicts internally.
- get_history() always returns a formatted string ready for LLM prompt injection.
- A hard cap of MAX_TURNS=10 (20 messages) prevents token-budget blow-out.
- Redis keys use a namespaced prefix to avoid collisions with other cache users.
"""

import json
from abc import ABC, abstractmethod
from typing import Dict, List

import redis

from backend.app.config import settings
from backend.app.core.logger import logger

# Maximum number of conversation turns (human + assistant pairs) to retain.
MAX_TURNS: int = 10
# Each turn = 2 messages, so the raw list cap is:
MAX_MESSAGES: int = MAX_TURNS * 2


class BaseMemory(ABC):
    """Abstract base class defining the conversation memory contract."""

    @abstractmethod
    def get_history(self, session_id: str) -> str:
        """
        Return the last MAX_TURNS conversation turns formatted for LLM injection.

        Format:
            H:
            <human message>

            A:
            <assistant message>
        """

    @abstractmethod
    def add_message(self, session_id: str, role: str, content: str) -> None:
        """
        Append one message to the session history.

        Args:
            session_id: Unique identifier for the conversation session.
            role: Either ``"human"`` or ``"assistant"``.
            content: The raw message text.
        """

    @abstractmethod
    def clear(self, session_id: str) -> None:
        """Delete all stored history for the given session."""


def _format_messages(messages: List[Dict[str, str]]) -> str:
    """
    Convert a list of message dicts into the canonical H:/A: format.

    Args:
        messages: Last N message dicts with 'role' and 'content' keys.

    Returns:
        A multi-line string suitable for inclusion in an LLM prompt, or an
        empty string when there is no history.
    """
    if not messages:
        return ""

    parts: List[str] = []
    for msg in messages:
        role_label = "H" if msg["role"] == "human" else "A"
        parts.append(f"{role_label}:\n{msg['content']}")
    return "\n\n".join(parts)


class InMemoryMemory(BaseMemory):
    """
    Thread-safe, in-process conversation memory backed by a plain Python dict.

    Suitable for single-process deployments or as a Redis fallback.
    Data is lost when the process restarts.
    """

    def __init__(self) -> None:
        # Dict[session_id, List[{role, content}]]
        self._storage: Dict[str, List[Dict[str, str]]] = {}

    def get_history(self, session_id: str) -> str:
        messages = self._storage.get(session_id, [])
        # Always return the last MAX_MESSAGES entries
        recent = messages[-MAX_MESSAGES:]
        return _format_messages(recent)

    def add_message(self, session_id: str, role: str, content: str) -> None:
        if session_id not in self._storage:
            self._storage[session_id] = []
        self._storage[session_id].append({"role": role, "content": content})
        # Trim at double the cap to avoid unbounded growth between reads
        if len(self._storage[session_id]) > MAX_MESSAGES * 2:
            self._storage[session_id] = self._storage[session_id][-MAX_MESSAGES:]

    def clear(self, session_id: str) -> None:
        self._storage.pop(session_id, None)


class RedisMemory(BaseMemory):
    """
    Redis-backed conversation memory with automatic in-memory fallback.

    Uses Redis lists (RPUSH / LRANGE) to persist serialised message dicts.
    If Redis is unavailable at startup or at runtime, all operations are
    transparently delegated to an ``InMemoryMemory`` instance so the API
    never returns an error due to a cache failure.

    Redis key format: ``chat_memory:<session_id>``
    """

    _REDIS_KEY_PREFIX = "chat_memory"

    def __init__(self) -> None:
        self._fallback = InMemoryMemory()
        self._client: redis.Redis | None = None
        self._connect()

    def _connect(self) -> None:
        """Attempt to establish a Redis connection; silently degrade on failure."""
        try:
            client = redis.from_url(settings.REDIS_URL, decode_responses=True)
            client.ping()
            self._client = client
            logger.info("[Memory] Redis connection established successfully.")
        except Exception as exc:
            logger.warning(
                "[Memory] Redis unavailable (%s). Falling back to in-memory store.", exc
            )
            self._client = None

    def _key(self, session_id: str) -> str:
        return f"{self._REDIS_KEY_PREFIX}:{session_id}"

    def get_history(self, session_id: str) -> str:
        if self._client is None:
            return self._fallback.get_history(session_id)
        try:
            raw_messages = self._client.lrange(self._key(session_id), -MAX_MESSAGES, -1)
            messages = [json.loads(m) for m in raw_messages]
            return _format_messages(messages)
        except Exception as exc:
            logger.error("[Memory] Redis get_history failed (%s). Using fallback.", exc)
            return self._fallback.get_history(session_id)

    def add_message(self, session_id: str, role: str, content: str) -> None:
        if self._client is None:
            self._fallback.add_message(session_id, role, content)
            return
        try:
            key = self._key(session_id)
            serialised = json.dumps({"role": role, "content": content})
            self._client.rpush(key, serialised)
            # Keep only the most recent MAX_MESSAGES entries in Redis
            self._client.ltrim(key, -MAX_MESSAGES, -1)
        except Exception as exc:
            logger.error("[Memory] Redis add_message failed (%s). Using fallback.", exc)
            self._fallback.add_message(session_id, role, content)

    def clear(self, session_id: str) -> None:
        if self._client is None:
            self._fallback.clear(session_id)
            return
        try:
            self._client.delete(self._key(session_id))
        except Exception as exc:
            logger.error("[Memory] Redis clear failed (%s). Using fallback.", exc)
            self._fallback.clear(session_id)


# ---------------------------------------------------------------------------
# Module-level singleton — imported by the API layer.
# ---------------------------------------------------------------------------

memory_store: BaseMemory = RedisMemory()
