import os
import sys
from pathlib import Path

# Add parent directory of backend to sys.path so we can import 'backend'
sys.path.append(str(Path(__file__).resolve().parent.parent))

from backend.app.core.logger import logger
from backend.app.database.ingest import ingest_orders_csv
from backend.app.vector.ingest import ingest_policies_pdfs

def main():
    logger.info("Starting complete dataset ingestion...")
    try:
        # Step 1: SQL orders database
        logger.info("Ingesting structured orders data...")
        ingest_orders_csv()
        
        # Step 2: Unstructured PDFs
        logger.info("Ingesting unstructured policy PDFs...")
        ingest_policies_pdfs()
        
        logger.info("Ingestion completed successfully!")
    except Exception as e:
        logger.critical(f"Ingestion pipeline failed: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
