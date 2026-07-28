import pytest
import sqlite3
import pandas as pd
from pathlib import Path
from backend.app.config import settings
from backend.app.database.db import get_db_connection, execute_sql_query, get_schema_info
from backend.app.database.ingest import ingest_orders_csv

def test_sqlite_ingestion():
    # Ingest the CSV
    ingest_orders_csv()
    
    # Assert database file exists
    assert settings.SQLITE_DB_PATH.exists()
    
    # Verify read-only connection
    conn = get_db_connection(read_only=True)
    assert isinstance(conn, sqlite3.Connection)
    
    # Verify schema contains the orders table
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='orders';")
    table = cursor.fetchone()
    assert table is not None
    assert table[0] == "orders"
    conn.close()

def test_sql_execution():
    # Ingest orders
    ingest_orders_csv()
    
    # Run test query
    query = "SELECT COUNT(*) as count FROM orders;"
    df = execute_sql_query(query)
    
    assert isinstance(df, pd.DataFrame)
    assert "count" in df.columns
    assert df.loc[0, "count"] > 0

def test_db_read_only_protection():
    # Ingest orders
    ingest_orders_csv()
    
    # Verify write commands fail under read-only connection
    query = "DELETE FROM orders WHERE order_id = 'ORD-1001';"
    with pytest.raises(Exception):
        execute_sql_query(query)
