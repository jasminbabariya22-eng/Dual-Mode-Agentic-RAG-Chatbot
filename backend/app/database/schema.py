import sqlite3
import hashlib
import threading
from typing import List, Dict, Set, Optional
from backend.app.config import settings
from backend.app.database.db import get_db_connection
from backend.app.core.logger import logger

class SchemaManager:
    """Manages introspection and caching of SQLite database schemas for Text-to-SQL generation."""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(SchemaManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
        
    def __init__(self):
        if getattr(self, "_initialized", False):
            return
            
        self.tables: List[str] = []
        self.columns: Dict[str, List[str]] = {}
        self.column_types: Dict[str, Dict[str, str]] = {}
        self.primary_keys: Dict[str, List[str]] = {}
        self.raw_schema_text: str = ""
        self.schema_version: str = "v1"
        self.refresh()
        self._initialized = True

    def refresh(self):
        """Introspects the SQLite database to fetch tables, columns, primary keys, and types."""
        conn = get_db_connection(read_only=True)
        cursor = conn.cursor()
        try:
            # 1. Fetch tables
            cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
            tables_data = cursor.fetchall()
            
            self.tables = []
            schema_parts = []
            schema_hash_inputs = []
            
            for t in tables_data:
                table_name = t['name']
                create_sql = t['sql']
                self.tables.append(table_name)
                
                # Fetch columns metadata
                cursor.execute(f"PRAGMA table_info({table_name});")
                cols = cursor.fetchall()
                
                self.columns[table_name] = []
                self.column_types[table_name] = {}
                self.primary_keys[table_name] = []
                
                col_defs = []
                for c in cols:
                    col_name = c['name']
                    col_type = c['type']
                    is_pk = c['pk'] > 0
                    
                    self.columns[table_name].append(col_name)
                    self.column_types[table_name][col_name] = col_type
                    if is_pk:
                        self.primary_keys[table_name].append(col_name)
                        
                    pk_indicator = " (PRIMARY KEY)" if is_pk else ""
                    col_defs.append(f"  - {col_name}: {col_type}{pk_indicator}")
                
                schema_hash_inputs.append(create_sql)
                
                # Format table definition for LLM prompt
                table_schema = f"Table: {table_name}\nColumns:\n" + "\n".join(col_defs)
                schema_parts.append(table_schema)
                
            self.raw_schema_text = "\n\n".join(schema_parts)
            
            # Compute a hash of the schemas as the schema version
            schema_combined = "".join(schema_hash_inputs)
            self.schema_version = hashlib.md5(schema_combined.encode('utf-8')).hexdigest()
            logger.info(f"Database schema introspected successfully. Schema Version: {self.schema_version}")
            
        except Exception as e:
            logger.error(f"Failed to introspect database schema: {str(e)}")
            raise e
        finally:
            conn.close()

    def get_schema(self) -> str:
        """Returns the introspected database schema string formatted for prompt injection."""
        return self.raw_schema_text

    def get_table_names(self) -> List[str]:
        """Returns a list of table names present in the database."""
        return self.tables

    def get_column_names(self, table: str) -> List[str]:
        """Returns a list of column names for a specific table."""
        return self.columns.get(table, [])

    def get_schema_version(self) -> str:
        """Returns MD5 hash representing the current schema catalog version."""
        return self.schema_version

# Global schema manager instance
schema_manager = SchemaManager()
