from prometheus_client import Counter, Gauge, Histogram, REGISTRY

# HTTP layer

HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests received.",
    labelnames=["method", "path", "status_code"],
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds.",
    labelnames=["method", "path"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)

# Workflow pipeline

WORKFLOW_DURATION_SECONDS = Histogram(
    "workflow_duration_seconds",
    "End-to-end LangGraph workflow latency in seconds.",
    labelnames=["route"],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0),
)

ROUTER_LATENCY_SECONDS = Histogram(
    "router_latency_seconds",
    "AgentRouter classification latency in seconds.",
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0),
)

RETRIEVAL_LATENCY_SECONDS = Histogram(
    "retrieval_latency_seconds",
    "HybridRetriever retrieval latency in seconds.",
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
)

SQL_GENERATION_LATENCY_SECONDS = Histogram(
    "sql_generation_latency_seconds",
    "Text-to-SQL generation latency in seconds.",
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0),
)

SQL_EXECUTION_LATENCY_SECONDS = Histogram(
    "sql_execution_latency_seconds",
    "SQLite query execution latency in seconds.",
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0),
)

LLM_LATENCY_SECONDS = Histogram(
    "llm_latency_seconds",
    "LLM answer generation latency in seconds.",
    buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 60.0, 120.0),
)

VECTOR_RETRIEVAL_COUNT = Histogram(
    "vector_retrieval_count",
    "Number of document chunks returned per retrieval call.",
    buckets=(1, 2, 3, 5, 10, 15, 20),
)

REQUESTS_BY_ROUTE_TOTAL = Counter(
    "requests_by_route_total",
    "Total workflow executions broken down by route.",
    labelnames=["route"],
)

# Cache

CACHE_HITS_TOTAL = Counter(
    "cache_hits_total",
    "Total cache hits.",
    labelnames=["cache_type"],   # "kv" | "semantic" | "prompt"
)

CACHE_MISSES_TOTAL = Counter(
    "cache_misses_total",
    "Total cache misses.",
    labelnames=["cache_type"],
)

# Errors

ERRORS_TOTAL = Counter(
    "errors_total",
    "Total errors broken down by component and type.",
    labelnames=["component", "error_type"],
)

# Connections / availability

REDIS_AVAILABLE = Gauge(
    "redis_available",
    "1 if Redis is reachable, 0 if using in-memory fallback.",
)

# Session / streaming

ACTIVE_SESSIONS_GAUGE = Gauge(
    "active_sessions_total",
    "Number of active chat sessions (approximation based on concurrent requests).",
)

STREAMING_REQUESTS_TOTAL = Counter(
    "streaming_requests_total",
    "Total streaming (SSE) requests received.",
)

# Convenience helpers used by the API layer

def record_workflow_metrics(execution_metrics: dict, route: str) -> None:
    """
    Extract per-stage latency figures from the workflow execution_metrics dict
    and observe them into the corresponding histograms.

    Called from the chat API endpoint after a successful workflow invocation.
    Converts millisecond values stored in the workflow state to seconds.

    Args:
        execution_metrics: The ``execution_metrics`` dict from AgentState.
        route:             The chosen route label (rag | sql | hybrid).
    """
    def _ms_to_s(key: str) -> float:
        return execution_metrics.get(key, 0.0) / 1000.0

    total_s = _ms_to_s("total_execution_time_ms")
    if total_s > 0:
        WORKFLOW_DURATION_SECONDS.labels(route=route).observe(total_s)

    router_s = _ms_to_s("router_time_ms")
    if router_s > 0:
        ROUTER_LATENCY_SECONDS.observe(router_s)

    retrieval_s = _ms_to_s("retrieval_time_ms")
    if retrieval_s > 0:
        RETRIEVAL_LATENCY_SECONDS.observe(retrieval_s)

    sql_gen_s = _ms_to_s("sql_generation_time_ms")
    if sql_gen_s > 0:
        SQL_GENERATION_LATENCY_SECONDS.observe(sql_gen_s)

    sql_exec_s = _ms_to_s("sql_execution_time_ms")
    if sql_exec_s > 0:
        SQL_EXECUTION_LATENCY_SECONDS.observe(sql_exec_s)

    llm_s = _ms_to_s("answer_generation_time_ms")
    if llm_s > 0:
        LLM_LATENCY_SECONDS.observe(llm_s)

    if route:
        REQUESTS_BY_ROUTE_TOTAL.labels(route=route).inc()
