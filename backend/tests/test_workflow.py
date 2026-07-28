import pytest
import json
from unittest.mock import AsyncMock, patch, MagicMock
from backend.app.agent.workflow import agent_workflow
from backend.app.agent.router import RouteDecision
from backend.app.database.text_to_sql import SQLExecutionResult

@pytest.mark.asyncio
async def test_workflow_rag_flow():
    """Verify that RAG route flow correctly fetches documents and answers queries."""
    # Create mock router
    mock_router = AsyncMock()
    mock_router.route.return_value = RouteDecision(mode="rag", confidence=0.99, reason="asks about policies")
    
    # Create mock retriever
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [{"content": "Leave duration is 15 days.", "citation": "hr_leave_policy.pdf (Page 1)"}]
    
    # Create mock LLM
    mock_llm_res = AsyncMock()
    mock_llm_res.content = "Employees get 15 days of leave. [leave_policy.pdf]"
    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value = mock_llm_res
    
    with patch("backend.app.agent.workflow.get_router", return_value=mock_router), \
         patch("backend.app.agent.workflow.get_retriever", return_value=mock_retriever), \
         patch("backend.app.agent.workflow.get_llm", return_value=mock_llm):
         
        initial = {
            "question": "What is the leave duration?",
            "route": "",
            "rag_context": "",
            "sql_query": "",
            "sql_result": "",
            "final_answer": "",
            "confidence": 0.0,
            "execution_metrics": {},
            "conversation_history": ""
        }
        res = await agent_workflow.ainvoke(initial)
        assert res["route"] == "rag"
        assert "Leave duration is 15 days." in res["rag_context"]
        assert res["final_answer"] == "Employees get 15 days of leave. [leave_policy.pdf]"
        assert res["execution_metrics"]["router_time_ms"] >= 0
        assert res["execution_metrics"]["retrieval_time_ms"] >= 0

@pytest.mark.asyncio
async def test_workflow_sql_flow():
    """Verify that SQL route flow generates, executes SQLite queries, and returns metrics."""
    # Create mock router
    mock_router = AsyncMock()
    mock_router.route.return_value = RouteDecision(mode="sql", confidence=0.98, reason="needs database sum")
    
    # Create mock SQL engine
    mock_sql_res = SQLExecutionResult(
        sql="SELECT COUNT(*) FROM orders",
        results=[{"count": 100}],
        success=True,
        errors=[],
        gen_latency_ms=15.0,
        exec_latency_ms=5.0
    )
    mock_engine = AsyncMock()
    mock_engine.execute_and_format.return_value = mock_sql_res
    
    # Create mock LLM
    mock_llm_res = AsyncMock()
    mock_llm_res.content = "There are 100 orders."
    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value = mock_llm_res
    
    with patch("backend.app.agent.workflow.get_router", return_value=mock_router), \
         patch("backend.app.agent.workflow.get_sql_engine", return_value=mock_engine), \
         patch("backend.app.agent.workflow.get_llm", return_value=mock_llm):
         
        initial = {
            "question": "Count total orders",
            "route": "",
            "rag_context": "",
            "sql_query": "",
            "sql_result": "",
            "final_answer": "",
            "confidence": 0.0,
            "execution_metrics": {},
            "conversation_history": ""
        }
        res = await agent_workflow.ainvoke(initial)
        assert res["route"] == "sql"
        assert res["sql_query"] == "SELECT COUNT(*) FROM orders"
        assert res["sql_result"]["results"][0]["count"]
        assert res["final_answer"] == "There are 100 orders."
        assert res["execution_metrics"]["sql_generation_time_ms"] == 15.0
        assert res["execution_metrics"]["sql_execution_time_ms"] == 5.0

@pytest.mark.asyncio
async def test_workflow_hybrid_flow():
    """Verify that Hybrid route executes RAG and SQL tasks concurrently and merges findings."""
    # Create mock router
    mock_router = AsyncMock()
    mock_router.route.return_value = RouteDecision(mode="hybrid", confidence=0.90, reason="analytical with documents")
    
    # Create mock retriever
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [{"content": "Laptop stand has 6 months warranty.", "citation": "warranty_policy.pdf"}]
    
    # Create mock SQL engine
    mock_sql_res = SQLExecutionResult(
        sql="SELECT COUNT(*) FROM orders WHERE product = 'Laptop Stand'",
        results=[{"count": 12}],
        success=True,
        errors=[],
        gen_latency_ms=20.0,
        exec_latency_ms=10.0
    )
    mock_engine = AsyncMock()
    mock_engine.execute_and_format.return_value = mock_sql_res
    
    # Create mock LLM
    mock_llm_res = AsyncMock()
    mock_llm_res.content = "We have 12 Laptop Stands, warranty is 6 months. [warranty_policy.pdf]"
    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value = mock_llm_res
    
    with patch("backend.app.agent.workflow.get_router", return_value=mock_router), \
         patch("backend.app.agent.workflow.get_retriever", return_value=mock_retriever), \
         patch("backend.app.agent.workflow.get_sql_engine", return_value=mock_engine), \
         patch("backend.app.agent.workflow.get_llm", return_value=mock_llm):
         
        initial = {
            "question": "Which Laptop Stands have pending orders?",
            "route": "",
            "rag_context": "",
            "sql_query": "",
            "sql_result": "",
            "final_answer": "",
            "confidence": 0.0,
            "execution_metrics": {},
            "conversation_history": ""
        }
        res = await agent_workflow.ainvoke(initial)
        assert res["route"] == "hybrid"
        assert "Laptop stand has 6 months warranty." in res["rag_context"]
        assert res["sql_query"] == "SELECT COUNT(*) FROM orders WHERE product = 'Laptop Stand'"
        assert res["sql_result"]["results"][0]["count"]
        assert "12 laptop stands" in res["final_answer"].lower()

@pytest.mark.asyncio
async def test_workflow_router_failure():
    """Verify that router failure gracefully defaults to RAG execution route."""
    # Create mock router that crashes
    mock_router = AsyncMock()
    mock_router.route.side_effect = ValueError("LLM Connection Refused")
    
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = []
    
    mock_llm_res = AsyncMock()
    mock_llm_res.content = "Mock RAG fallback answer. [rag.pdf]"
    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value = mock_llm_res
    
    with patch("backend.app.agent.workflow.get_router", return_value=mock_router), \
         patch("backend.app.agent.workflow.get_retriever", return_value=mock_retriever), \
         patch("backend.app.agent.workflow.get_llm", return_value=mock_llm):
         
        initial = {
            "question": "Maternity leave",
            "route": "",
            "rag_context": "",
            "sql_query": "",
            "sql_result": "",
            "final_answer": "",
            "confidence": 0.0,
            "execution_metrics": {},
            "conversation_history": ""
        }
        res = await agent_workflow.ainvoke(initial)
        assert res["route"] == "rag"
        assert res["confidence"] == 0.50

@pytest.mark.asyncio
async def test_workflow_retriever_failure():
    """Verify that retriever errors do not crash the pipeline and log errors safely."""
    # Create mock router
    mock_router = AsyncMock()
    mock_router.route.return_value = RouteDecision(mode="rag", confidence=0.95, reason="policy check")
    
    # Create mock retriever that fails
    mock_retriever = MagicMock()
    mock_retriever.retrieve.side_effect = Exception("ChromaDB Index not ready")
    
    mock_llm_res = AsyncMock()
    mock_llm_res.content = "Failed retrieve. [N/A]"
    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value = mock_llm_res
    
    with patch("backend.app.agent.workflow.get_router", return_value=mock_router), \
         patch("backend.app.agent.workflow.get_retriever", return_value=mock_retriever), \
         patch("backend.app.agent.workflow.get_llm", return_value=mock_llm):
         
        initial = {
            "question": "Leave policy",
            "route": "",
            "rag_context": "",
            "sql_query": "",
            "sql_result": "",
            "final_answer": "",
            "confidence": 0.0,
            "execution_metrics": {},
            "conversation_history": ""
        }
        res = await agent_workflow.ainvoke(initial)
        assert res["route"] == "rag"
        assert "Error during document retrieval." in res["rag_context"]

@pytest.mark.asyncio
async def test_workflow_sql_failure():
    """Verify that SQL generation/execution failures are safely captured in the SQL result."""
    # Create mock router
    mock_router = AsyncMock()
    mock_router.route.return_value = RouteDecision(mode="sql", confidence=0.95, reason="db check")
    
    # Create mock SQL engine that fails
    mock_engine = AsyncMock()
    mock_engine.execute_and_format.side_effect = Exception("Database locked")
    
    mock_llm_res = AsyncMock()
    mock_llm_res.content = "Database execution failed."
    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value = mock_llm_res
    
    with patch("backend.app.agent.workflow.get_router", return_value=mock_router), \
         patch("backend.app.agent.workflow.get_sql_engine", return_value=mock_engine), \
         patch("backend.app.agent.workflow.get_llm", return_value=mock_llm):
         
        initial = {
            "question": "Show all orders",
            "route": "",
            "rag_context": "",
            "sql_query": "",
            "sql_result": "",
            "final_answer": "",
            "confidence": 0.0,
            "execution_metrics": {},
            "conversation_history": ""
        }
        res = await agent_workflow.ainvoke(initial)
        assert res["route"] == "sql"
        assert isinstance(res["sql_result"], dict)
        assert "error" in res["sql_result"]
        assert "Database locked" in res["sql_result"]["error"]

@pytest.mark.asyncio
async def test_workflow_llm_failure():
    """Verify that final answer composer handles LLM failures gracefully."""
    # Create mock router
    mock_router = AsyncMock()
    mock_router.route.return_value = RouteDecision(mode="rag", confidence=0.95, reason="policy check")
    
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = []
    
    # Create mock LLM that fails
    mock_llm = AsyncMock()
    mock_llm.ainvoke.side_effect = RuntimeError("Out of quota limit")
    
    with patch("backend.app.agent.workflow.get_router", return_value=mock_router), \
         patch("backend.app.agent.workflow.get_retriever", return_value=mock_retriever), \
         patch("backend.app.agent.workflow.get_llm", return_value=mock_llm):
         
        initial = {
            "question": "Leave policy",
            "route": "",
            "rag_context": "",
            "sql_query": "",
            "sql_result": "",
            "final_answer": "",
            "confidence": 0.0,
            "execution_metrics": {},
            "conversation_history": ""
        }
        res = await agent_workflow.ainvoke(initial)

        assert res["final_answer"] == (
            "I apologize, but I encountered an internal error while generating your response. [N/A]"
        )
