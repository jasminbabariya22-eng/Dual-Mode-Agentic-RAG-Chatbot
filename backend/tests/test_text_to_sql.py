import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from backend.app.database.schema import schema_manager
from backend.app.database.text_to_sql import (
    SQLValidator, 
    TextToSQLEngine, 
    SQLValidationException,
    parse_relative_dates,
    sanitize_sql,
    SQLGenerationResponse,
    SQLExecutionResult
)

def test_schema_extraction():
    """Verify that schema manager correctly introspects SQLite schema attributes."""
    schema_manager.refresh()
    tables = schema_manager.get_table_names()
    assert "orders" in tables
    
    cols = schema_manager.get_column_names("orders")
    assert "order_id" in cols
    assert "customer" in cols
    assert "amount" in cols
    assert "status" in cols
    assert "order_date" in cols
    
    assert schema_manager.get_schema_version() is not None

def test_sql_validator_safe_selects():
    """Verify that safe SELECT queries pass validation successfully."""
    validator = SQLValidator()
    
    # Simple SELECT
    res = validator.validate("SELECT * FROM orders;")
    assert res.is_valid
    
    # Aggregation & grouping
    res = validator.validate("SELECT product, SUM(amount) AS revenue FROM orders GROUP BY product ORDER BY revenue DESC;")
    assert res.is_valid

def test_sql_validator_unsafe_queries():
    """Verify that unsafe/write/structural alteration statements are blocked."""
    validator = SQLValidator()
    
    # Block DELETE
    assert not validator.validate("DELETE FROM orders;").is_valid
    assert not validator.validate("DELETE FROM orders WHERE order_id = '123';").is_valid
    
    # Block DROP TABLE
    assert not validator.validate("DROP TABLE orders;").is_valid
    
    # Block INSERT, UPDATE, ALTER, TRUNCATE
    assert not validator.validate("UPDATE orders SET status = 'shipped';").is_valid
    assert not validator.validate("INSERT INTO orders (order_id) VALUES ('123');").is_valid
    assert not validator.validate("ALTER TABLE orders ADD COLUMN test TEXT;").is_valid
    
    # Block Union query injection
    assert not validator.validate("SELECT * FROM orders UNION SELECT * FROM other;").is_valid
    
    # Block semicolons & comments
    assert not validator.validate("SELECT * FROM orders; -- comment").is_valid
    assert not validator.validate("SELECT * FROM orders; DROP TABLE orders;").is_valid

def test_relative_date_parsing():
    """Verify that relative date phrases are parsed correctly into instruction rules."""
    ctx = parse_relative_dates("How many orders were placed last month?")
    assert "2026-05-01" in ctx
    assert "2026-05-31" in ctx
    
    ctx = parse_relative_dates("How many orders were placed this month?")
    assert "2026-06-01" in ctx
    assert "2026-06-30" in ctx
    
    ctx = parse_relative_dates("Orders placed today?")
    assert "2026-06-15" in ctx
    
    ctx = parse_relative_dates("Orders placed yesterday?")
    assert "2026-06-14" in ctx
    
    ctx = parse_relative_dates("Orders placed last 30 days?")
    assert "2026-05-16" in ctx
    assert "2026-06-15" in ctx

def test_sql_sanitization():
    """Verify that SQL sanitization normals are applied correctly."""
    # Markdown block stripping
    assert sanitize_sql("```sql\nSELECT * FROM orders;\n```") == "SELECT * FROM orders"
    # Semicolon stripping
    assert sanitize_sql("SELECT * FROM orders;") == "SELECT * FROM orders"
    # Quote normalization
    assert sanitize_sql('SELECT * FROM orders WHERE status = "pending";') == "SELECT * FROM orders WHERE status = 'pending'"
    # Whitespace normalization
    assert sanitize_sql("SELECT   COUNT(*)   FROM   orders") == "SELECT COUNT(*) FROM orders"

@pytest.mark.asyncio
async def test_sql_generation_flow_cache_miss_and_hit():
    """Verify the Text-to-SQL generation pipeline, mock LLM execution, cache miss, and cache hit flows."""
    engine = TextToSQLEngine()
    
    # We will mock cache_manager methods
    mock_kv = {}
    
    def mock_get(prefix, key):
        return mock_kv.get(f"{prefix}:{key}")
        
    def mock_set(prefix, key, val, ttl=None):
        mock_kv[f"{prefix}:{key}"] = val
        
    with patch('backend.app.core.cache.cache_manager.get_kv', side_effect=mock_get), \
         patch('backend.app.core.cache.cache_manager.set_kv', side_effect=mock_set):
         
        # Make a mock LLM output
        mock_response = AsyncMock()
        mock_response.content = "SELECT COUNT(*) FROM orders WHERE status = 'pending';"
        
        llm_class_path = f"{engine.llm.__class__.__module__}.{engine.llm.__class__.__name__}"
        with patch(f"{llm_class_path}.ainvoke", return_value=mock_response) as mock_ainvoke:
            # 1. First execution: Cache MISS
            res1 = await engine.generate_sql("Count pending orders")
            assert isinstance(res1, SQLGenerationResponse)
            assert res1.sql == "SELECT COUNT(*) FROM orders WHERE status = 'pending'"
            assert not res1.cached
            mock_ainvoke.assert_called_once()
            
            # 2. Second execution: Cache HIT (LLM invoke should NOT be called again)
            mock_ainvoke.reset_mock()
            res2 = await engine.generate_sql("Count pending orders")
            assert isinstance(res2, SQLGenerationResponse)
            assert res2.sql == "SELECT COUNT(*) FROM orders WHERE status = 'pending'"
            assert res2.cached
            mock_ainvoke.assert_not_called()
