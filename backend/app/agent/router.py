import json
import time
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from backend.app.config import settings
from backend.app.core.logger import logger
from backend.app.core.llm import get_llm
from backend.app.database.schema import schema_manager

class RouteDecision(BaseModel):
    """Pydantic model representing the routing decision choice made by the Agent Router."""
    mode: Literal["rag", "sql", "hybrid"] = Field(description="The execution route choice: rag, sql, or hybrid.")
    confidence: float = Field(description="Confidence score of the routing decision (between 0.0 and 1.0).")
    reason: str = Field(description="The reasoning explanation supporting this routing selection.")

class AgentRouter:
    """Class to intelligently route user queries to RAG, SQL, or hybrid query execution paths."""
    
    def __init__(self):
        self.llm = get_llm()
        self.prompt_template = PromptTemplate(
            input_variables=["schema_info", "question"],
            template="""You are an intelligent query router for a Dual-Mode Agentic RAG Chatbot.
Your task is to classify an incoming user question into one of three execution modes:

1. "rag": For questions about company policies, documentation, FAQs, warranties, refunds, leave policies, or general knowledge.
2. "sql": For analytical questions that require querying structured database tables (e.g., counting orders, calculating revenues, checking order status, listing order details).
3. "hybrid": For complex questions that require both document knowledge (RAG) and database details (SQL). (e.g., finding orders for products under warranty, checking if a policy applies to pending orders).

DATABASE SCHEMA FOR REFERENCE:
{schema_info}

CRITICAL RULES:
- Return ONLY a valid JSON object matching the schema below.
- Do NOT wrap JSON in code blocks (e.g. do NOT use ```json).
- Never write SQL, do not answer the user question, only classify it.

JSON SCHEMA:
{{
    "mode": "rag" | "sql" | "hybrid",
    "confidence": float (between 0.0 and 1.0),
    "reason": "Detailed explanation of the choice"
}}

User Question: {question}
JSON Output:"""
        )

    def _heuristic_fallback(self, question: str) -> RouteDecision:
        """Fallback rule-based heuristic classifier used when LLM JSON parsing crashes."""
        lower_q = question.lower()
        
        # Define keyword lists
        sql_indicators = [
            "count", "how many", "average", "avg", "sum", "total", "revenue", 
            "orders", "placed", "sales", "transaction", "amount", "customer", 
            "status", "pending", "shipped", "delivered"
        ]
        rag_indicators = [
            "warranty", "leave", "faq", "policy", "return", "refund", 
            "pricing", "discount", "sick", "vacation", "maternity", "paternity",
            "support", "contact", "help"
        ]
        
        has_sql = any(indicator in lower_q for indicator in sql_indicators)
        has_rag = any(indicator in lower_q for indicator in rag_indicators)
        
        if has_sql and has_rag:
            return RouteDecision(
                mode="hybrid",
                confidence=0.70,
                reason="Heuristic Fallback: Detected indicators for both SQL tables and Policy documentation."
            )
        elif has_sql:
            return RouteDecision(
                mode="sql",
                confidence=0.75,
                reason="Heuristic Fallback: Detected database keywords (orders/sales/customers/status)."
            )
        elif has_rag:
            return RouteDecision(
                mode="rag",
                confidence=0.75,
                reason="Heuristic Fallback: Detected policy documentation keywords (warranty/leave/return)."
            )
        else:
            return RouteDecision(
                mode="rag",
                confidence=0.50,
                reason="Heuristic Fallback: Defaulting to RAG for unrecognized request pattern."
            )

    async def route(self, question: str) -> RouteDecision:
        """Determines execution route (RAG, SQL, or hybrid) for a user question."""
        start_time = time.perf_counter()
        
        schema_info = schema_manager.get_schema()
        prompt = self.prompt_template.format(
            schema_info=schema_info,
            question=question
        )
        
        logger.info(f"[Agent Router] Classifying route for question: '{question}'")
        
        try:
            response = await self.llm.ainvoke(prompt)
            raw_text = response.content if hasattr(response, 'content') else str(response)
            
            # Clean raw output from code blocks
            cleaned = raw_text.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            
            data = json.loads(cleaned)
            
            # Extract attributes
            mode = data.get("mode", "rag").lower()
            if mode not in ["rag", "sql", "hybrid"]:
                mode = "rag"
                
            confidence = float(data.get("confidence", 0.8))
            reason = data.get("reason", "Successfully routed via LLM classification.")
            
            decision = RouteDecision(mode=mode, confidence=confidence, reason=reason)
            
        except Exception as e:
            logger.error(f"[Agent Router] JSON parsing failed: {str(e)}. Triggering heuristic fallback.")
            decision = self._heuristic_fallback(question)
            
        latency = (time.perf_counter() - start_time) * 1000
        logger.info(
            f"[Agent Router Decision] Latency={latency:.1f}ms | Chosen Route={decision.mode.upper()} | "
            f"Confidence={decision.confidence:.2f} | Reason='{decision.reason}'"
        )
        
        return decision
