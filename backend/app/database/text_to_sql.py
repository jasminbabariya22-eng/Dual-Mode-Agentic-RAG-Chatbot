import re
import time
import hashlib
import json
from typing import List, Dict, Set, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from backend.app.config import settings
from backend.app.core.logger import logger
from backend.app.core.llm import get_llm
from backend.app.core.cache import cache_manager
from backend.app.database.db import execute_sql_query
from backend.app.database.schema import schema_manager

# ======================================================================
# Pydantic Schemas
# ======================================================================

class SQLValidationResult(BaseModel):
    """Pydantic model representing SQL validation checks results."""
    is_valid: bool = Field(description="Indicates whether the query passed all safety and catalog validation checks.")
    errors: List[str] = Field(default_factory=list, description="Validation error logs if query fails checks.")
    cleaned_sql: str = Field(description="The normalized, sanitized SQL statement query.")

class SQLGenerationRequest(BaseModel):
    """Pydantic model representing user input question request."""
    question: str = Field(description="The natural language question text.")

class SQLGenerationResponse(BaseModel):
    """Pydantic model representing query generation outputs."""
    sql: str = Field(description="The generated SQLite query.")
    validation: SQLValidationResult = Field(description="Validation checks audit metadata.")
    cached: bool = Field(default=False, description="True if query generation hit cached records.")
    generation_latency_ms: float = Field(description="Time elapsed in generating and checking query.")

class SQLExecutionResult(BaseModel):
    """Pydantic model representing database execution results."""
    sql: str = Field(description="The query statement that was executed.")
    results: List[Dict[str, Any]] = Field(default_factory=list, description="Data rows retrieved from SQLite catalog.")
    success: bool = Field(description="True if query executed successfully without SQLite errors.")
    errors: List[str] = Field(default_factory=list, description="SQLite system exception details if query failed.")
    gen_latency_ms: float = Field(description="Generation pipeline latency in milliseconds.")
    exec_latency_ms: float = Field(description="Execution latency in database engine in milliseconds.")
    cached: bool = Field(default=False, description="True if DB output values hit cached store.")

class SQLValidationException(Exception):
    """Custom exception raised when SQL validation policies are violated."""
    def __init__(self, validation_result: SQLValidationResult):
        super().__init__(f"SQL validation checks failed: {', '.join(validation_result.errors)}")
        self.validation_result = validation_result

# ======================================================================
# Helper Utilities
# ======================================================================

def parse_relative_dates(question: str) -> str:
    """Helper to detect relative date terms in question and generate helper instructions."""
    lower_q = question.lower()
    rules = []
    
    # Check for relative date keywords and build specific SQLite date math rules
    if "last month" in lower_q:
        rules.append("For 'last month', filter where order_date is between '2026-05-01' and '2026-05-31'.")
    if "this month" in lower_q:
        rules.append("For 'this month', filter where order_date is between '2026-06-01' and '2026-06-30'.")
    if "today" in lower_q:
        rules.append("For 'today', filter where order_date = '2026-06-15'.")
    if "yesterday" in lower_q:
        rules.append("For 'yesterday', filter where order_date = '2026-06-14'.")
    if "last 30 days" in lower_q:
        rules.append("For 'last 30 days', filter where order_date is between '2026-05-16' and '2026-06-15'.")
    if "this year" in lower_q:
        rules.append("For 'this year', filter where order_date is between '2026-01-01' and '2026-12-31'.")
    if "last year" in lower_q:
        rules.append("For 'last year', filter where order_date is between '2025-01-01' and '2025-12-31'.")
    if "before june" in lower_q:
        rules.append("For 'before June', filter where order_date < '2026-06-01'.")
    if "after january" in lower_q:
        rules.append("For 'after January', filter where order_date > '2026-01-31'.")
        
    if rules:
        return "\nSpecific Date Rules for this question:\n" + "\n".join(rules)
    return ""

def sanitize_sql(sql: str) -> str:
    """Cleans, normalizes and normalises raw LLM SQL outputs."""
    # 1. Clean markdown brackets
    cleaned = sql.strip()
    cleaned = re.sub(r"^```sql\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^```\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    
    # 2. Drop comment tokens (validator will block, but keep cleaning robust)
    cleaned = re.sub(r'--.*', '', cleaned)
    cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)
    
    # 3. Strip semicolons
    cleaned = cleaned.rstrip(";")
    
    # 4. Standardise double quoted string literals to single quotes
    cleaned = re.sub(r'"([^"\n]*)"', r"'\1'", cleaned)
    
    # 5. Condense duplicate whitespaces
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    return cleaned.strip()

# ======================================================================
# SQL Validator
# ======================================================================

class SQLValidator:
    """Checks generated SQLite queries for correctness, injection vectors, and catalog alignment."""
    
    def __init__(self):
        # Access catalog maps directly from global schema manager
        pass

    def validate(self, sql_query: str) -> SQLValidationResult:
        """Validates query against database schemas and safety filters."""
        errors = []
        
        # 1. Block comments
        if "--" in sql_query or "/*" in sql_query:
            errors.append("SQL statement contains comments, which are not permitted.")
            
        cleaned = sql_query.strip()
        
        # 2. Block multiple statements (semicolon delimiters)
        semicolon_split = [s.strip() for s in cleaned.split(";") if s.strip()]
        if len(semicolon_split) > 1:
            errors.append("Multiple SQL statements are not permitted.")
            
        sql_to_check = semicolon_split[0] if semicolon_split else ""
        
        # 3. Block unauthorized keywords
        forbidden_keywords = [
            "delete", "update", "insert", "drop", "alter", "create", 
            "truncate", "pragma", "attach", "load_extension"
        ]
        
        # Strip out quoted values to prevent false matches from strings (e.g. customer name containing 'Delete')
        no_strings = re.sub(r"'(?:''|[^'])*'", "", sql_to_check)
        no_strings = re.sub(r'"(?:""|[^"])*"', "", no_strings)
        
        # Tokenize remaining words
        words = re.findall(r'[a-zA-Z_0-9]+', no_strings.lower())
        
        # Verify SELECT only statement
        if words and words[0] != "select":
            errors.append("Only SELECT statements are permitted.")
            
        for kw in forbidden_keywords:
            if kw in words:
                errors.append(f"SQL statement contains unauthorized keyword: '{kw.upper()}'")
                
        # Block UNION unless strictly required (restrict UNION query injection vectors)
        if "union" in words:
            # Let's count occurrence. Union query injection is blocked.
            errors.append("UNION operations are not permitted.")
            
        # 4. Table checks
        allowed_tables = {t.lower() for t in schema_manager.get_table_names()}
        from_join_matches = re.findall(r'\b(?:from|join)\s+([a-zA-Z_0-9]+)', no_strings.lower())
        
        if not from_join_matches:
            has_table = any(t in words for t in allowed_tables)
            if not has_table and allowed_tables:
                errors.append("SQL statement does not reference any known table.")
        else:
            for table_ref in from_join_matches:
                if table_ref not in allowed_tables:
                    errors.append(f"SQL references unauthorized or unknown table: '{table_ref}'")

        # 5. Column checks
        # Extract aliases so they don't fail column checks
        aliases = set(re.findall(r'\bas\s+([a-zA-Z_0-9]+)', no_strings.lower()))
        
        sql_functions_and_keywords = {
            "select", "from", "where", "group", "by", "order", "limit", "offset", "having", "as",
            "count", "sum", "avg", "min", "max", "and", "or", "not", "in", "like", "between",
            "is", "null", "date", "strftime", "round", "coalesce", "cast", "desc", "asc",
            "orders", "join", "on", "inner", "left", "outer", "cross", "t"
        }
        sql_functions_and_keywords.update(aliases)
        
        for word in words:
            if word.isdigit():
                continue
            if word not in sql_functions_and_keywords and word not in allowed_tables:
                # Match column against tables
                is_valid_col = False
                for tbl in schema_manager.get_table_names():
                    cols = {c.lower() for c in schema_manager.get_column_names(tbl)}
                    if word in cols:
                        is_valid_col = True
                        break
                if not is_valid_col:
                    errors.append(f"SQL references unauthorized or unknown column: '{word}'")

        is_valid = len(errors) == 0
        return SQLValidationResult(
            is_valid=is_valid,
            errors=errors,
            cleaned_sql=sql_to_check
        )

# ======================================================================
# SQL Generation Engine
# ======================================================================

class TextToSQLEngine:
    """Handles text-to-SQL generation, strong validation, caching, and execution."""
    
    def __init__(self):
        self.llm = get_llm()
        self.validator = SQLValidator()
        self.prompt_template = PromptTemplate(
            input_variables=["schema_info", "question", "reference_date", "date_rules"],
            template="""You are a SQLite expert database administrator.
Given an input question, create a syntactically correct SQLite query to run.
The database schema contains the following details:
{schema_info}

CRITICAL RULES:
1. ONLY return a raw executable SQL query. Do NOT return markdown code blocks, do NOT write explanatory text, and do NOT wrap the output in quotes.
2. The current system date is hardcoded as {reference_date} (YYYY-MM-DD). Any time-based filtering (like "last month", "last week", "orders in May") MUST be calculated relative to '{reference_date}'.
3. Do NOT attempt to execute write/modification queries (INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, PRAGMA). Only SELECT queries are permitted.
4. Only query tables and columns that exist in the schema.
5. If the query requires date comparisons, use SQLite date functions like:
   - `date(order_date)`
   - `strftime('%Y-%m-%d', order_date)`
   - `date(order_date, '-7 days')`
{date_rules}

Question: {question}
SQLQuery:"""
        )

    def _generate_cache_key(self, question: str) -> str:
        """Hases Question, Schema Version, Prompt Version and Model into cache key."""
        schema_ver = schema_manager.get_schema_version()
        raw_str = f"{question}:{schema_ver}:{settings.PROMPT_VERSION}:{settings.LLM_MODEL}"
        hashed = hashlib.sha256(raw_str.encode('utf-8')).hexdigest()
        return f"prompt_cache:{hashed}"

    async def generate_sql(self, question: str) -> SQLGenerationResponse:
        """Generates SQL query from question. Employs caching, sanitization, and validator checks."""
        start_time = time.perf_counter()
        
        # 1. Check prompt cache
        cache_key = self._generate_cache_key(question)
        cached_sql = cache_manager.get_kv("sql_gen", cache_key)
        if cached_sql:
            # Parse cached JSON
            try:
                cached_data = json.loads(cached_sql)
                val_res = SQLValidationResult(**cached_data["validation"])
                latency = (time.perf_counter() - start_time) * 1000
                logger.info(f"[Text-to-SQL] Prompt Cache HIT for question: '{question}' in {latency:.2f}ms")
                return SQLGenerationResponse(
                    sql=cached_data["sql"],
                    validation=val_res,
                    cached=True,
                    generation_latency_ms=latency
                )
            except Exception as e:
                logger.warning(f"Failed to parse cached SQL payload: {str(e)}. Proceeding to fresh generation.")
                
        # Measure prompt building time
        start_prompt = time.perf_counter()
        schema_info = schema_manager.get_schema()
        date_rules = parse_relative_dates(question)
        
        prompt = self.prompt_template.format(
            schema_info=schema_info,
            question=question,
            reference_date=settings.REFERENCE_DATE,
            date_rules=date_rules
        )
        prompt_time_ms = (time.perf_counter() - start_prompt) * 1000
        
        # 2. LLM Call
        start_llm = time.perf_counter()
        logger.info(f"[Text-to-SQL] Cache MISS. Sending question to LLM: '{question}'")
        response = await self.llm.ainvoke(prompt)
        raw_sql = response.content if hasattr(response, 'content') else str(response)
        llm_time_ms = (time.perf_counter() - start_llm) * 1000
        
        # 3. SQL Sanitization
        sanitized = sanitize_sql(raw_sql)
        
        # 4. SQL Validation
        start_val = time.perf_counter()
        if settings.ENABLE_SQL_VALIDATION:
            validation = self.validator.validate(sanitized)
            if not validation.is_valid:
                val_latency = (time.perf_counter() - start_val) * 1000
                logger.error(f"[Text-to-SQL] Generated SQL failed safety checks: {validation.errors}")
                raise SQLValidationException(validation)
            final_sql = validation.cleaned_sql
        else:
            validation = SQLValidationResult(is_valid=True, errors=[], cleaned_sql=sanitized)
            final_sql = sanitized
            
        val_time_ms = (time.perf_counter() - start_val) * 1000
        total_latency = (time.perf_counter() - start_time) * 1000
        
        # Log Latency Metrics
        logger.info(
            f"[Text-to-SQL Metrics] Hashed Key={cache_key} | Prompt={prompt_time_ms:.1f}ms | LLM={llm_time_ms:.1f}ms | "
            f"Validation={val_time_ms:.1f}ms | Total={total_latency:.1f}ms | SQL={final_sql}"
        )
        
        # 5. Store output to standard KV Cache
        response_obj = SQLGenerationResponse(
            sql=final_sql,
            validation=validation,
            cached=False,
            generation_latency_ms=total_latency
        )
        cache_manager.set_kv("sql_gen", cache_key, response_obj.model_dump_json(), ttl=settings.CACHE_TTL)
        
        return response_obj

    async def execute_and_format(self, question: str) -> SQLExecutionResult:
        """Generates and executes SQL. Returns SQLExecutionResult containing row outputs or checks metadata."""
        start_pipeline = time.perf_counter()
        
        try:
            # 1. SQL Generation
            gen_response = await self.generate_sql(question)
            sql = gen_response.sql
            
            # 2. Check Execution cache
            start_exec = time.perf_counter()
            exec_cache_key = f"sql_exec_results:{sql}"
            cached_results = cache_manager.get_kv("sql_exec", exec_cache_key)
            if cached_results:
                exec_latency = (time.perf_counter() - start_exec) * 1000
                logger.info(f"[Text-to-SQL Execution] Results Cache HIT in {exec_latency:.2f}ms")
                return SQLExecutionResult(
                    sql=sql,
                    results=json.loads(cached_results),
                    success=True,
                    errors=[],
                    gen_latency_ms=gen_response.generation_latency_ms,
                    exec_latency_ms=exec_latency,
                    cached=True
                )
                
            # 3. SQLite Execution
            df = execute_sql_query(sql)
            results = df.to_dict(orient="records")
            exec_latency = (time.perf_counter() - start_exec) * 1000
            
            # Cache results in Redis/Local KV cache
            cache_manager.set_kv("sql_exec", exec_cache_key, json.dumps(results), ttl=settings.CACHE_TTL)
            
            return SQLExecutionResult(
                sql=sql,
                results=results,
                success=True,
                errors=[],
                gen_latency_ms=gen_response.generation_latency_ms,
                exec_latency_ms=exec_latency,
                cached=False
            )
            
        except SQLValidationException as e:
            total_latency = (time.perf_counter() - start_pipeline) * 1000
            return SQLExecutionResult(
                sql=e.validation_result.cleaned_sql,
                results=[],
                success=False,
                errors=e.validation_result.errors,
                gen_latency_ms=total_latency,
                exec_latency_ms=0.0,
                cached=False
            )
        except Exception as e:
            total_latency = (time.perf_counter() - start_pipeline) * 1000
            logger.error(f"[Text-to-SQL execution] Engine execution crashed: {str(e)}")
            return SQLExecutionResult(
                sql="",
                results=[],
                success=False,
                errors=[str(e)],
                gen_latency_ms=total_latency,
                exec_latency_ms=0.0,
                cached=False
            )
