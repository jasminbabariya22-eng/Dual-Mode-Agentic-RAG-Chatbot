import csv
import pandas as pd
from pathlib import Path
from backend.app.config import settings
from backend.app.core.logger import logger
from backend.app.database.db import get_db_connection

def ingest_orders_csv():
    """Reads orders.csv or orders.xlsx from the Dataset directory and writes to SQLite orders table."""
    dataset_dir = settings.DATASET_DIR
    csv_path = dataset_dir / "orders.csv"
    xlsx_path = dataset_dir / "orders.xlsx"
    
    if csv_path.exists():
        logger.info(f"Found orders.csv at {csv_path}, reading...")
        df = pd.read_csv(csv_path)
    elif xlsx_path.exists():
        logger.info(f"Found orders.xlsx at {xlsx_path}, reading...")
        df = pd.read_excel(xlsx_path)
    else:
        raise FileNotFoundError(f"Neither orders.csv nor orders.xlsx was found in {dataset_dir}")

    # Standardize column names (strip whitespace)
    df.columns = [col.strip() for col in df.columns]
    
    # Strip whitespace from string columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip()

    logger.info(f"Loaded {len(df)} orders from dataset file.")

    # Write to SQLite
    db_path = settings.SQLITE_DB_PATH
    # Ensure parent dir exists
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    conn = get_db_connection(read_only=False)
    cursor = conn.cursor()
    try:
        # Create table with explicit schema
        cursor.execute("DROP TABLE IF EXISTS orders;")
        cursor.execute("""
            CREATE TABLE orders (
                order_id TEXT PRIMARY KEY,
                customer TEXT NOT NULL,
                product TEXT NOT NULL,
                amount INTEGER NOT NULL,
                status TEXT NOT NULL,
                order_date TEXT NOT NULL
            );
        """)
        
        # Load into table
        df.to_sql('orders', conn, if_exists='append', index=False)
        
        # Create Indexes for fast querying
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_orders_customer ON orders(customer);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_orders_order_date ON orders(order_date);")
        
        conn.commit()
        logger.info("Orders database created and indexes added successfully.")
        
    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to ingest orders: {str(e)}")
        raise e
    finally:
        conn.close()

if __name__ == "__main__":
    ingest_orders_csv()
