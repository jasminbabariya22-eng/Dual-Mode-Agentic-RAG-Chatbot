import pytest
from pathlib import Path
from backend.app.config import settings
from backend.app.vector.store import VectorStoreManager
from backend.app.vector.ingest import ingest_policies_pdfs

def test_vector_ingestion():
    # Ingest the PDFs
    ingest_policies_pdfs()
    
    # Assert vector database directory was created
    assert settings.CHROMA_DB_PATH.exists()
    
    # Verify we can query the vector store
    vstore = VectorStoreManager()
    results = vstore.similarity_search("What is the sick leave policy?", top_k=2)
    
    assert len(results) > 0
    assert "content" in results[0]
    assert "metadata" in results[0]
    assert "source" in results[0]["metadata"]
    assert "page" in results[0]["metadata"]
