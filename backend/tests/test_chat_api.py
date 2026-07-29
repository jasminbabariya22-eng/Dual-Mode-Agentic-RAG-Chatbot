import asyncio
import json
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport

from backend.main import app
from backend.app.memory.memory import InMemoryMemory, _format_messages, MAX_TURNS, MAX_MESSAGES

# Fixtures

# Shared default workflow result used across multiple tests
_DEFAULT_WORKFLOW_RESULT: Dict[str, Any] = {
    "question": "What is the warranty period?",
    "route": "rag",
    "rag_context": "Document chunk:\nLaptop warranty is 12 months.\nSource: warranty_policy.pdf (Page 2)",
    "sql_query": "",
    "sql_result": "",
    "final_answer": "The laptop warranty period is 12 months.",
    "confidence": 0.95,
    "execution_metrics": {
        "router_time_ms": 10.0,
        "retrieval_time_ms": 50.0,
        "sql_generation_time_ms": 0.0,
        "sql_execution_time_ms": 0.0,
        "answer_generation_time_ms": 200.0,
        "total_execution_time_ms": 260.0,
    },
    "conversation_history": "",
    "stream_queue": None,
}

@pytest.fixture
def client() -> TestClient:
    """Synchronous TestClient for non-streaming tests."""
    return TestClient(app, raise_server_exceptions=False)

@pytest.fixture
def mock_workflow_rag():
    """Patch agent_workflow.ainvoke with a successful RAG result."""
    result = dict(_DEFAULT_WORKFLOW_RESULT)
    with patch("backend.app.api.chat.agent_workflow") as mock_wf:
        mock_wf.ainvoke = AsyncMock(return_value=result)
        yield mock_wf, result

@pytest.fixture
def mock_workflow_sql():
    """Patch agent_workflow.ainvoke with a successful SQL result."""
    result = {
        **_DEFAULT_WORKFLOW_RESULT,
        "route": "sql",
        "rag_context": "",
        "sql_query": "SELECT COUNT(*) as cnt FROM orders",
        "sql_result": json.dumps([{"cnt": 42}]),
        "final_answer": "There are 42 orders in total.",
        "confidence": 0.97,
    }
    with patch("backend.app.api.chat.agent_workflow") as mock_wf:
        mock_wf.ainvoke = AsyncMock(return_value=result)
        yield mock_wf, result

@pytest.fixture
def mock_workflow_hybrid():
    """Patch agent_workflow.ainvoke with a successful Hybrid result."""
    result = {
        **_DEFAULT_WORKFLOW_RESULT,
        "route": "hybrid",
        "rag_context": "Document chunk:\nWarranty is 6 months.\nSource: warranty_policy.pdf",
        "sql_query": "SELECT * FROM orders WHERE product = 'Laptop Stand'",
        "sql_result": json.dumps([{"order_id": "ORD-001", "status": "Pending"}]),
        "final_answer": "Laptop Stands have a 6-month warranty and order ORD-001 is Pending.",
        "confidence": 0.88,
    }
    with patch("backend.app.api.chat.agent_workflow") as mock_wf:
        mock_wf.ainvoke = AsyncMock(return_value=result)
        yield mock_wf, result

@pytest.fixture
def mock_memory():
    """Patch the module-level memory_store in chat.py with an InMemoryMemory."""
    mem = InMemoryMemory()
    with patch("backend.app.api.chat.memory_store", new=mem):
        yield mem

# POST /api/v1/chat — Normal chat / route tests

def test_chat_rag_route(client: TestClient, mock_workflow_rag, mock_memory):
    """Happy path: RAG route returns sources, empty SQL fields."""
    mock_wf, result = mock_workflow_rag
    response = client.post(
        "/api/v1/chat",
        json={"question": "What is the warranty period?"},
    )

    assert response.status_code == 200
    body = response.json()

    assert body["answer"] == "The laptop warranty period is 12 months."
    assert body["route"] == "rag"
    assert body["confidence"] == pytest.approx(0.95)
    assert "warranty_policy.pdf (Page 2)" in body["sources"]
    assert body["sql_query"] == ""
    assert body["sql_result"] == []
    assert "request_id" in body
    assert "session_id" in body
    assert body["execution_metrics"]["router_time_ms"] == pytest.approx(10.0)

def test_chat_sql_route(client: TestClient, mock_workflow_sql, mock_memory):
    """Happy path: SQL route returns sql_query and sql_result rows."""
    mock_wf, result = mock_workflow_sql
    response = client.post(
        "/api/v1/chat",
        json={"question": "How many orders do we have?"},
    )

    assert response.status_code == 200
    body = response.json()

    assert body["route"] == "sql"
    assert body["sql_query"] == "SELECT COUNT(*) as cnt FROM orders"
    assert body["sql_result"] == [{"cnt": 42}]
    assert body["sources"] == []
    assert body["answer"] == "There are 42 orders in total."

def test_chat_hybrid_route(client: TestClient, mock_workflow_hybrid, mock_memory):
    """Happy path: Hybrid route returns both sources and SQL result."""
    mock_wf, result = mock_workflow_hybrid
    response = client.post(
        "/api/v1/chat",
        json={"question": "Which Laptop Stands under warranty have pending orders?"},
    )

    assert response.status_code == 200
    body = response.json()

    assert body["route"] == "hybrid"
    assert "warranty_policy.pdf" in body["sources"]
    assert body["sql_query"] == "SELECT * FROM orders WHERE product = 'Laptop Stand'"
    assert body["sql_result"] == [{"order_id": "ORD-001", "status": "Pending"}]

# POST /api/v1/chat — Session ID handling

def test_chat_generates_session_id_when_missing(client: TestClient, mock_workflow_rag, mock_memory):
    """When no session_id is supplied a UUID is auto-generated."""
    response = client.post("/api/v1/chat", json={"question": "Hello"})

    assert response.status_code == 200
    body = response.json()
    # Must be a valid UUID4
    sid = body["session_id"]
    parsed = uuid.UUID(sid, version=4)
    assert str(parsed) == sid

def test_chat_uses_provided_session_id(client: TestClient, mock_workflow_rag, mock_memory):
    """When a session_id is supplied it is echoed back unchanged."""
    response = client.post(
        "/api/v1/chat",
        json={"question": "Hello", "session_id": "my-custom-session"},
    )

    assert response.status_code == 200
    assert response.json()["session_id"] == "my-custom-session"

# POST /api/v1/chat — Input validation errors

def test_chat_rejects_empty_question(client: TestClient):
    """A blank question must return 422 Validation Error."""
    response = client.post("/api/v1/chat", json={"question": ""})
    assert response.status_code == 422

def test_chat_rejects_whitespace_only_question(client: TestClient):
    """A whitespace-only question must return 422 Validation Error."""
    response = client.post("/api/v1/chat", json={"question": "   "})
    assert response.status_code == 422

def test_chat_rejects_missing_question_field(client: TestClient):
    """Missing 'question' field must return 422 Validation Error."""
    response = client.post("/api/v1/chat", json={})
    assert response.status_code == 422

def test_chat_rejects_question_exceeding_max_length(client: TestClient):
    """A question longer than 2000 characters must return 422 Validation Error."""
    response = client.post("/api/v1/chat", json={"question": "x" * 2001})
    assert response.status_code == 422

# POST /api/v1/chat — Workflow exception → HTTP 500

def test_chat_returns_500_on_workflow_exception(client: TestClient, mock_memory):
    """An unhandled workflow exception must surface as HTTP 500."""
    with patch("backend.app.api.chat.agent_workflow") as mock_wf:
        mock_wf.ainvoke = AsyncMock(side_effect=RuntimeError("Ollama connection refused"))

        response = client.post("/api/v1/chat", json={"question": "What is the policy?"})

    assert response.status_code == 500

# POST /api/v1/chat — Memory persistence

def test_chat_persists_turn_to_memory(client: TestClient, mock_workflow_rag, mock_memory):
    """After a successful chat, human + assistant turns are saved to memory."""
    mock_wf, result = mock_workflow_rag
    session_id = "persist-test-session"

    client.post(
        "/api/v1/chat",
        json={"question": "What is the warranty?", "session_id": session_id},
    )

    history = mock_memory.get_history(session_id)
    assert "What is the warranty?" in history
    assert "The laptop warranty period is 12 months." in history

def test_chat_injects_history_into_state(client: TestClient, mock_memory):
    """On the second turn, conversation history is injected into workflow state."""
    session_id = "history-injection-session"
    # Pre-populate memory with one turn
    mock_memory.add_message(session_id, "human", "First question about leave.")
    mock_memory.add_message(session_id, "assistant", "Leave is 15 days.")

    captured_state: Dict = {}

    async def capture_invoke(state):
        captured_state.update(state)
        return dict(_DEFAULT_WORKFLOW_RESULT)

    with patch("backend.app.api.chat.agent_workflow") as mock_wf:
        mock_wf.ainvoke = AsyncMock(side_effect=capture_invoke)
        client.post(
            "/api/v1/chat",
            json={"question": "What about sick leave?", "session_id": session_id},
        )

    assert "First question about leave." in captured_state.get("conversation_history", "")
    assert "Leave is 15 days." in captured_state.get("conversation_history", "")

# POST /api/v1/chat/stream — SSE streaming

@pytest.mark.asyncio
async def test_chat_stream_yields_tokens():
    """The streaming endpoint should emit token events and a final done event."""

    async def fake_invoke(state):
        """Simulate the workflow pushing tokens into the queue."""
        queue = state.get("stream_queue")
        if queue:
            await queue.put("Hello ")
            await queue.put("World!")
            await queue.put(None)  # sentinel
        result = dict(_DEFAULT_WORKFLOW_RESULT)
        result["stream_queue"] = queue
        return result

    mem = InMemoryMemory()

    with patch("backend.app.api.chat.agent_workflow") as mock_wf, \
         patch("backend.app.api.chat.memory_store", new=mem):
        mock_wf.ainvoke = AsyncMock(side_effect=fake_invoke)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            async with ac.stream(
                "POST",
                "/api/v1/chat/stream",
                json={"question": "What is the warranty?"},
            ) as response:
                assert response.status_code == 200
                assert "text/event-stream" in response.headers["content-type"]

                events: List[str] = []
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        events.append(line[len("data: "):])

    # Must have token events
    token_events = [e for e in events if e != "[DONE]" and "token" in e]
    assert len(token_events) >= 1

    # Must end with [DONE]
    assert events[-1] == "[DONE]"

    # Tokens must reconstruct the answer
    tokens = [json.loads(e)["token"] for e in token_events]
    assert "".join(tokens) == "Hello World!"

@pytest.mark.asyncio
async def test_chat_stream_persists_memory_after_done():
    """Memory must be saved after the full stream completes."""

    async def fake_invoke(state):
        queue = state.get("stream_queue")
        if queue:
            await queue.put("Streamed answer text.")
            await queue.put(None)
        result = dict(_DEFAULT_WORKFLOW_RESULT)
        result["stream_queue"] = queue
        return result

    mem = InMemoryMemory()
    session_id = "stream-memory-session"

    with patch("backend.app.api.chat.agent_workflow") as mock_wf, \
         patch("backend.app.api.chat.memory_store", new=mem):
        mock_wf.ainvoke = AsyncMock(side_effect=fake_invoke)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            async with ac.stream(
                "POST",
                "/api/v1/chat/stream",
                json={"question": "Tell me about returns", "session_id": session_id},
            ) as response:
                # Consume the entire stream
                async for _ in response.aiter_lines():
                    pass

    history = mem.get_history(session_id)
    assert "Tell me about returns" in history
    assert "Streamed answer text." in history

@pytest.mark.asyncio
async def test_chat_stream_validation_error():
    """Empty question to the stream endpoint should return 422."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post(
            "/api/v1/chat/stream",
            json={"question": ""},
        )
    assert response.status_code == 422

# Memory unit tests

class TestInMemoryMemory:
    """Unit tests for InMemoryMemory."""

    def test_empty_history_returns_empty_string(self):
        mem = InMemoryMemory()
        assert mem.get_history("s1") == ""

    def test_add_and_retrieve_single_turn(self):
        mem = InMemoryMemory()
        mem.add_message("s1", "human", "Hello?")
        mem.add_message("s1", "assistant", "Hi there!")
        history = mem.get_history("s1")
        assert "H:\nHello?" in history
        assert "A:\nHi there!" in history

    def test_history_respects_max_turns_cap(self):
        mem = InMemoryMemory()
        # Add more than MAX_TURNS turns
        for i in range(MAX_TURNS + 5):
            mem.add_message("s1", "human", f"Q{i}")
            mem.add_message("s1", "assistant", f"A{i}")

        history = mem.get_history("s1")
        # Only the last MAX_TURNS turns should appear
        # The earliest turns should be evicted
        assert "Q0" not in history
        assert f"Q{MAX_TURNS + 4}" in history

    def test_clear_removes_session(self):
        mem = InMemoryMemory()
        mem.add_message("s1", "human", "Test")
        mem.clear("s1")
        assert mem.get_history("s1") == ""

    def test_clear_nonexistent_session_does_not_raise(self):
        mem = InMemoryMemory()
        mem.clear("nonexistent")  # Should not raise

class TestFormatMessages:
    """Unit tests for the _format_messages helper."""

    def test_empty_list_returns_empty_string(self):
        assert _format_messages([]) == ""

    def test_human_labelled_as_H(self):
        result = _format_messages([{"role": "human", "content": "Hello?"}])
        assert result == "H:\nHello?"

    def test_assistant_labelled_as_A(self):
        result = _format_messages([{"role": "assistant", "content": "Hi!"}])
        assert result == "A:\nHi!"

    def test_multiple_turns_joined_with_blank_lines(self):
        messages = [
            {"role": "human", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "human", "content": "Q2"},
        ]
        result = _format_messages(messages)
        assert result == "H:\nQ1\n\nA:\nA1\n\nH:\nQ2"
