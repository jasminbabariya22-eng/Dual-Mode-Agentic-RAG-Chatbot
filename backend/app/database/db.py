import sqlite3
import pandas as pd
from pathlib import Path
from backend.app.config import settings
from backend.app.core.logger import logger

def get_db_connection(read_only: bool = True) -> sqlite3.Connection:
    """Get connection to the SQLite database. Supports read-only mode to prevent write exploits."""
    db_path = settings.SQLITE_DB_PATH
    
    if read_only:
        # SQLite read-only connection requires uri=True and mode=ro
        if not db_path.exists():
            raise FileNotFoundError(f"Database file not found at {db_path}. Please run ingestion first.")
        conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    else:
        conn = sqlite3.connect(db_path)
    
    conn.row_factory = sqlite3.Row
    return conn

def execute_sql_query(query: str) -> pd.DataFrame:
    """Executes a SQL query safely in read-only mode and returns a pandas DataFrame."""
    conn = get_db_connection(read_only=True)
    try:
        logger.info(f"Executing SQL Query: {query}")
        df = pd.read_sql_query(query, conn)
        return df
    except Exception as e:
        logger.error(f"Failed to execute query: {query}. Error: {str(e)}")
        raise e
    finally:
        conn.close()

def get_schema_info() -> str:
    """Returns the schema description and sample rows for LLM text-to-SQL generation."""
    conn = get_db_connection(read_only=True)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        tables = cursor.fetchall()
        schema_text = []
        for table in tables:
            table_name = table['name']
            create_sql = table['sql']
            schema_text.append(f"Table: {table_name}\nSchema:\n{create_sql}")
            
            # Fetch sample rows
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 3;")
            sample_rows = cursor.fetchall()
            if sample_rows:
                sample_str = "Sample Rows:\n" + "\n".join([str(dict(row)) for row in sample_rows])
                schema_text.append(sample_str)
            schema_text.append("-" * 40)
        return "\n".join(schema_text)
    except Exception as e:
        logger.error(f"Error fetching schema: {str(e)}")
        raise e
    finally:
        conn.close()
