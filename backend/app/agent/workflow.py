import time
import json
import asyncio
from functools import lru_cache
from typing import TypedDict, Dict, Any, List, Optional
from langgraph.graph import StateGraph, START, END
from backend.app.config import settings
from backend.app.core.logger import logger
from backend.app.core.llm import get_llm
from backend.app.agent.router import AgentRouter
from backend.app.vector.retriever import HybridRetriever
from backend.app.database.text_to_sql import TextToSQLEngine
from backend.app.core.guardrails import (
    validate_question,
    validate_sql,
    detect_prompt_injection,
    detect_jailbreak,
    validate_output,
    validate_citations,
    verify_answer_against_context
)

# AgentState Definition

class AgentState(TypedDict):
    # This state keeps track of everything as we pass it around the nodes
    question: str
    route: str
    rag_context: str
    sql_query: str
    sql_result: Any
    final_answer: str
    confidence: float
    execution_metrics: dict
    conversation_history: str
    stream_queue: Optional[Any]

# Prompt Templates

ANSWER_PROMPT_TEMPLATE = """You are a helpful, expert customer support assistant for a Dual-Mode Agentic RAG Chatbot.
Your task is to generate a comprehensive, accurate, and markdown-formatted answer to the user's question.

CONVERSATION HISTORY:
{conversation_history}

User Question: {question}

CONTEXT PROVIDED:
---
RAG Document Context:
{rag_context}
---
SQL Query: {sql_query}
SQL Result Data:
{sql_result}
---

CRITICAL GUIDELINES:
1. ONLY answer based on the provided context (documents and/or SQL database results). Do NOT use external knowledge.
2. If RAG Context is empty or insufficient to answer the question, explicitly say: "I do not have enough policy documentation context to answer this query."
3. If SQL Query was run and SQL Result Data is empty or contains no records, explicitly say: "No matching database records were found."
4. If this is a hybrid query, combine the SQL data facts (e.g. order statuses, amounts) with the policy document details (e.g. warranty lengths, return options) to compose a unified answer.
5. Never hallucinate any facts, order IDs, or policy rules.
6. Format your final response in clean, professional Markdown.

Final Answer:"""

# Shared Component Instantiation

@lru_cache()
def get_router() -> AgentRouter:
    return AgentRouter()

@lru_cache()
def get_retriever() -> HybridRetriever:
    return HybridRetriever()

@lru_cache()
def get_sql_engine() -> TextToSQLEngine:
    return TextToSQLEngine()

# Heuristics & Execution Helpers

def _extract_content(response: Any) -> str:
    # Digs through the messy LLM response to pull out the plain text we need
    if response is None:
        logger.warning("[LLM] Response is None — no content to extract.")
        return ""

    # Plain string (some simple wrappers or mocks)
    if isinstance(response, str):
        return response

    # Standard LangChain AIMessage / BaseMessage
    if hasattr(response, "content"):
        content = response.content
        if isinstance(content, str):
            return content
        # Some multimodal models return a list of content blocks
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict):
                    parts.append(block.get("text", ""))
                elif hasattr(block, "text"):
                    parts.append(str(block.text))
            return "".join(parts)
        return str(content)

    # ChatResult / LLMResult (older API)
    if hasattr(response, "generations"):
        try:
            return response.generations[0][0].text
        except (IndexError, AttributeError):
            pass

    # Dict response (some custom wrappers)
    if isinstance(response, dict):
        for key in ("content", "text", "message", "answer", "output"):
            if key in response and isinstance(response[key], str):
                return response[key]

    # Last resort — convert whatever we received
    logger.warning(
        "[LLM] Unrecognised response type '%s' — coercing to string.",
        type(response).__name__,
    )
    return str(response)

async def run_rag_retrieval(question: str) -> Dict[str, Any]:
    # Look up the question in our PDF documents to see what the company policies say
    start_time = time.perf_counter()
    retriever = get_retriever()
    try:
        # Offload synchronous retrieval to a thread to avoid blocking event loop
        docs = await asyncio.to_thread(retriever.retrieve, question)
        
        context_parts = []
        for doc in docs:
            content = doc.get("content", "")
            citation = doc.get("citation", "Unknown")
            context_parts.append(f"Document chunk:\n{content}\nSource: {citation}")
        rag_context = "\n\n".join(context_parts)
        latency = (time.perf_counter() - start_time) * 1000
        return {
            "rag_context": rag_context,
            "retrieval_time_ms": latency
        }
    except Exception as e:
        logger.error(f"[Workflow Helper] Retrieval task crashed: {str(e)}")
        return {
            "rag_context": "Error during document retrieval.",
            "retrieval_time_ms": 0.0
        }

async def run_sql_execution(question: str) -> Dict[str, Any]:
    # Convert the user's question into SQL and run it against our database
    sql_engine = get_sql_engine()
    try:
        # execute_and_format is async, so we await directly
        res = await sql_engine.execute_and_format(question)
        
        # Guardrail SQL validation
        if res.success:
            validate_sql(res.sql)
            
        return {
            "sql_query": res.sql,
            "sql_result": {
                    "results": res.results
                } if res.success else {
                    "error": ", ".join(res.errors)
                },
            "sql_generation_time_ms": res.gen_latency_ms,
            "sql_execution_time_ms": res.exec_latency_ms
        }
    except Exception as e:
        logger.error(f"[Workflow Helper] SQL execution task crashed: {str(e)}")
        return {
            "sql_query": "",
            "sql_result": {"error": f"Exception: {str(e)}"},
            "sql_generation_time_ms": 0.0,
            "sql_execution_time_ms": 0.0
        }

# Graph Nodes

async def router_node(state: AgentState) -> Dict[str, Any]:
    # This node figures out if we need to search PDFs, run SQL, or do both!
    start_time = time.perf_counter()
    metrics = state.get("execution_metrics", {})
    if not metrics:
        metrics = {
            "workflow_start_time": start_time,
            "router_time_ms": 0.0,
            "retrieval_time_ms": 0.0,
            "sql_generation_time_ms": 0.0,
            "sql_execution_time_ms": 0.0,
            "answer_generation_time_ms": 0.0,
            "total_execution_time_ms": 0.0
        }
    else:
        # Fallback if somehow metrics exist but missing start time
        if "workflow_start_time" not in metrics:
            metrics["workflow_start_time"] = start_time
    
    router = get_router()
    try:
        decision = await router.route(state["question"])
        route_val = decision.mode
        confidence_val = decision.confidence
    except Exception as e:
        logger.error("[Workflow Node] Router failed. Defaulting to RAG.")
        route_val = "rag"
        confidence_val = 0.50
        
    latency = (time.perf_counter() - start_time) * 1000
    metrics["router_time_ms"] = latency
    
    return {
        "route": route_val,
        "confidence": confidence_val,
        "execution_metrics": metrics
    }

async def rag_node(state: AgentState) -> Dict[str, Any]:
    """Node responsible for retrieving contextual reference policy files."""
    metrics = state.get("execution_metrics", {})
    res = await run_rag_retrieval(state["question"])
    metrics["retrieval_time_ms"] = res["retrieval_time_ms"]
    
    return {
        "rag_context": res["rag_context"],
        "execution_metrics": metrics
    }

async def sql_node(state: AgentState) -> Dict[str, Any]:
    """Node responsible for generating SQLite queries and fetching rows."""
    metrics = state.get("execution_metrics", {})
    res = await run_sql_execution(state["question"])
    
    metrics["sql_generation_time_ms"] = res["sql_generation_time_ms"]
    metrics["sql_execution_time_ms"] = res["sql_execution_time_ms"]
    
    return {
        "sql_query": res["sql_query"],
        "sql_result": res["sql_result"],
        "execution_metrics": metrics
    }

async def hybrid_node(state: AgentState) -> Dict[str, Any]:
    """Concurrently executes RAG retrieval and SQL query execution nodes and merges state."""
    metrics = state.get("execution_metrics", {})
    
    # Concurrent execution for performance optimization
    rag_task = asyncio.create_task(run_rag_retrieval(state["question"]))
    sql_task = asyncio.create_task(run_sql_execution(state["question"]))
    
    rag_res, sql_res = await asyncio.gather(rag_task, sql_task)
    
    metrics["retrieval_time_ms"] = rag_res["retrieval_time_ms"]
    metrics["sql_generation_time_ms"] = sql_res["sql_generation_time_ms"]
    metrics["sql_execution_time_ms"] = sql_res["sql_execution_time_ms"]
    
    return {
        "rag_context": rag_res["rag_context"],
        "sql_query": sql_res["sql_query"],
        "sql_result": sql_res["sql_result"],
        "execution_metrics": metrics
    }

async def answer_node(state: AgentState) -> Dict[str, Any]:
    # Generates the final response based on context collected in State
    start_time = time.perf_counter()
    metrics = state.get("execution_metrics", {})

    llm = get_llm()

    question = state["question"]
    conversation_history = state.get("conversation_history", "")
    rag_context = state.get("rag_context", "")
    sql_query = state.get("sql_query", "")
    
    # Format sql_result as JSON string for the prompt
    raw_sql_result = state.get("sql_result", "")
    if not isinstance(raw_sql_result, str):
        sql_result = (
            raw_sql_result
            if isinstance(raw_sql_result, str)
            else json.dumps(raw_sql_result, indent=2)
        )
    else:
        sql_result = raw_sql_result

    # Diagnostic prompt-building log
    logger.info(
        "[Answer Node] Building prompt | question_len=%d context_len=%d "
        "history_len=%d sql_query_len=%d",
        len(question),
        len(rag_context),
        len(conversation_history),
        len(sql_query),
    )

    try:
        prompt = ANSWER_PROMPT_TEMPLATE.format(
            question=question,
            conversation_history=conversation_history,
            rag_context=rag_context,
            sql_query=sql_query,
            sql_result=sql_result,
        )
    except KeyError:
        # Should not happen with fixed template, but guard defensively
        logger.exception("[Answer Node] Prompt template format failed: %s")
        prompt = (
            f"Answer the following question based on the provided context.\n"
            f"Question: {question}\nContext: {rag_context}"
        )

    logger.info(
        "[Answer Node] Prompt ready | total_chars=%d | invoking LLM=%s",
        len(prompt),
        type(llm).__name__,
    )
    
    # Temporarily log full prompt for debugging
    logger.debug(
            "[Answer Node] Prompt length=%d characters",
            len(prompt),
        )

    queue = state.get("stream_queue")
    answer = ""

    try:
        if queue and settings.LLM_STREAMING and hasattr(llm, "astream"):
            # Streaming path (POST /chat/stream)
            full_content: list[str] = []
            logger.info("[Answer Node] Starting streaming (astream)...")
            try:
                async for chunk in llm.astream(prompt):
                    token = _extract_content(chunk)
                    if token:
                        full_content.append(token)
                        await queue.put(token)
                await queue.put(None)  # Signal end-of-stream
                answer = "".join(full_content)
                logger.info(
                    "[Answer Node] Streaming complete | tokens=%d answer_len=%d",
                    len(full_content),
                    len(answer),
                )
            except Exception:
                logger.warning(
                    "[Answer Node] astream() failed — falling back to ainvoke()",
                    exc_info=True,
                )
                logger.info("Calling LLM synchronously fallback...")
                response = await llm.ainvoke(prompt)
                logger.info("LLM call completed. Response type: %s", type(response))
                logger.info("Response: %r", response)
                answer = _extract_content(response)
                logger.info(
                    "[Answer Node] ainvoke() fallback produced answer_len=%d",
                    len(answer),
                )
                await queue.put(answer)
                await queue.put(None)
        else:
            # Standard synchronous path (POST /chat)
            logger.info("[Answer Node] Calling llm.ainvoke()")
            response = await llm.ainvoke(prompt)
            logger.info("LLM call completed.")
            logger.info(
                "[Answer Node] Raw LLM response type: %s",
                type(response).__name__,
            )
            # logger.info("Response: %r", response)
            answer = _extract_content(response)
            logger.info(
                "[Answer Node] Answer extracted | len=%d | preview='%s'",
                len(answer),
                answer[:120].replace("\n", " "),
            )

        # Empty response guard
        if not answer.strip():
            logger.error("[Answer Node] LLM returned empty response.")
            answer = "I couldn't generate a response based on the provided context. [N/A]"
            if queue:
                await queue.put(answer)
                await queue.put(None)

    except Exception:
        logger.exception("[Answer Node] Answer generation failed")

        answer = (
            "I apologize, but I encountered an internal error while generating your response. [N/A]"
        )

        if queue:
            await queue.put(answer)
            await queue.put(None)
        
    latency = (time.perf_counter() - start_time) * 1000
    metrics["answer_generation_time_ms"] = latency
    
    # Calculate complete graph workflow wall-clock time
    workflow_start = metrics.get("workflow_start_time", start_time)
    total_time = (time.perf_counter() - workflow_start) * 1000
    metrics["total_execution_time_ms"] = total_time
    
    logger.info(f"[Workflow Nodes Audit] Answer composed in {latency:.1f}ms. Total wall execution={total_time:.1f}ms")
    
    return {
        "final_answer": answer,
        "execution_metrics": metrics
    }

# Graph Routing Decision Logic

def route_decision(state: AgentState) -> str:
    """Dynamic routing mapper based on classification results stored in state."""
    r = state.get("route", "rag").lower()
    if r in ["rag", "sql", "hybrid"]:
        return r
    return "rag"

# Guardrail Nodes

def input_validator_node(state: AgentState) -> dict:
    validate_question(state["question"])
    return {}

def prompt_injection_node(state: AgentState) -> dict:
    detect_prompt_injection(state["question"])
    return {}

def jailbreak_node(state: AgentState) -> dict:
    detect_jailbreak(state["question"])
    return {}

def output_validator_node(state: AgentState) -> dict:
    validate_output(state["final_answer"])
    return {}

def citation_validator_node(state: AgentState) -> dict:
    validate_citations(state["final_answer"], state.get("route", "rag"))
    return {}

def hallucination_checker_node(state: AgentState) -> dict:
    # Await checking logic
    result = verify_answer_against_context(state["question"], state.get("rag_context", ""), state["final_answer"], state.get("confidence", 1.0))
    # Update confidence if hallucination detected
    new_confidence = result.confidence
    
    updates = {"confidence": new_confidence}
    
    if new_confidence < settings.MIN_CONFIDENCE:
        updates["final_answer"] = "I don't have enough reliable information to answer this question."
        
    return updates

def pydantic_validation_node(state: AgentState) -> dict:
    # Just validate through the Pydantic schema GuardedAnswer
    # We don't change state, just ensure it parses
    from backend.app.guardrails.response_schema import GuardedAnswer
    # Assuming sources are extracted somewhere, or we pass an empty list if not
    GuardedAnswer(
        answer=state["final_answer"],
        citations=[],  # Ideally extracted, but we ensure schema is met
        confidence=state.get("confidence", 1.0),
        route=state.get("route", "rag"),
        validated=True,
        hallucination_score=1.0
    )
    return {}

# LangGraph Workflow Construction

workflow_graph = StateGraph(AgentState)

# Add Node mapping structures
workflow_graph.add_node("input_validator", input_validator_node)
workflow_graph.add_node("prompt_injection", prompt_injection_node)
workflow_graph.add_node("jailbreak", jailbreak_node)
workflow_graph.add_node("router", router_node)
workflow_graph.add_node("rag", rag_node)
workflow_graph.add_node("sql", sql_node)
workflow_graph.add_node("hybrid", hybrid_node)
workflow_graph.add_node("answer", answer_node)
workflow_graph.add_node("output_validator", output_validator_node)
workflow_graph.add_node("citation_validator", citation_validator_node)
workflow_graph.add_node("hallucination_checker", hallucination_checker_node)
workflow_graph.add_node("pydantic_validation", pydantic_validation_node)

# Set starting point
workflow_graph.set_entry_point("input_validator")

# Input guardrail edges
workflow_graph.add_edge("input_validator", "prompt_injection")
workflow_graph.add_edge("prompt_injection", "jailbreak")
workflow_graph.add_edge("jailbreak", "router")

# Add conditional edges
workflow_graph.add_conditional_edges(
    "router",
    route_decision,
    {
        "rag": "rag",
        "sql": "sql",
        "hybrid": "hybrid"
    }
)

# Connect intermediate nodes to composer node
workflow_graph.add_edge("rag", "answer")
workflow_graph.add_edge("sql", "answer")
workflow_graph.add_edge("hybrid", "answer")

# Output guardrail edges
workflow_graph.add_edge("answer", "output_validator")
workflow_graph.add_edge("output_validator", "citation_validator")
workflow_graph.add_edge("citation_validator", "hallucination_checker")
workflow_graph.add_edge("hallucination_checker", "pydantic_validation")

# Set final endpoint
workflow_graph.add_edge("pydantic_validation", END)

# Compile application workflow
agent_workflow = workflow_graph.compile()
