import pytest
from unittest.mock import AsyncMock, patch
from backend.app.agent.router import AgentRouter, RouteDecision

@pytest.mark.asyncio
async def test_router_policy_question_rag():
    """Verify routing of documentation and policy queries to RAG."""
    router = AgentRouter()
    
    mock_response = AsyncMock()
    mock_response.content = '{"mode": "rag", "confidence": 0.95, "reason": "Query asks about leave policy details."}'
    
    llm_class_path = f"{router.llm.__class__.__module__}.{router.llm.__class__.__name__}"
    with patch(f"{llm_class_path}.ainvoke", return_value=mock_response):
        decision = await router.route("What is the maternity leave policy?")
        assert decision.mode == "rag"
        assert decision.confidence == 0.95
        assert "leave" in decision.reason

@pytest.mark.asyncio
async def test_router_analytical_question_sql():
    """Verify routing of database metrics queries to SQL."""
    router = AgentRouter()
    
    mock_response = AsyncMock()
    mock_response.content = '{"mode": "sql", "confidence": 0.98, "reason": "Requires order count calculations."}'
    
    llm_class_path = f"{router.llm.__class__.__module__}.{router.llm.__class__.__name__}"
    with patch(f"{llm_class_path}.ainvoke", return_value=mock_response):
        decision = await router.route("How many pending orders were placed in May?")
        assert decision.mode == "sql"
        assert decision.confidence == 0.98

@pytest.mark.asyncio
async def test_router_mixed_question_hybrid():
    """Verify routing of cross-references queries to Hybrid."""
    router = AgentRouter()
    
    mock_response = AsyncMock()
    mock_response.content = '{"mode": "hybrid", "confidence": 0.90, "reason": "Needs to cross-reference product warranties with pending orders database."}'
    
    llm_class_path = f"{router.llm.__class__.__module__}.{router.llm.__class__.__name__}"
    with patch(f"{llm_class_path}.ainvoke", return_value=mock_response):
        decision = await router.route("Which products under warranty have pending orders?")
        assert decision.mode == "hybrid"
        assert decision.confidence == 0.90

@pytest.mark.asyncio
async def test_router_unknown_question():
    """Verify routing fallback for generic questions."""
    router = AgentRouter()
    
    mock_response = AsyncMock()
    mock_response.content = '{"mode": "rag", "confidence": 0.50, "reason": "Irrelevant generic question."}'
    
    llm_class_path = f"{router.llm.__class__.__module__}.{router.llm.__class__.__name__}"
    with patch(f"{llm_class_path}.ainvoke", return_value=mock_response):
        decision = await router.route("What is the capital of France?")
        assert decision.mode == "rag"
        assert decision.confidence == 0.50

@pytest.mark.asyncio
async def test_router_malformed_output_fallback():
    """Verify that parsing failure gracefully falls back to heuristic rules."""
    router = AgentRouter()
    
    mock_response = AsyncMock()
    # Malformed text output failing JSON checks
    mock_response.content = "Malformed response text from LLM, definitely not valid JSON."
    
    llm_class_path = f"{router.llm.__class__.__module__}.{router.llm.__class__.__name__}"
    with patch(f"{llm_class_path}.ainvoke", return_value=mock_response):
        # 1. Test policy keywords fallback
        decision = await router.route("What is the return policy details?")
        assert decision.mode == "rag"
        assert "Fallback" in decision.reason
        
        # 2. Test database keywords fallback
        decision = await router.route("How many orders?")
        assert decision.mode == "sql"
        assert "Fallback" in decision.reason
        
        # 3. Test mixed fallback
        decision = await router.route("What is the refund status of pending orders?")
        assert decision.mode == "hybrid"
        assert "Fallback" in decision.reason
