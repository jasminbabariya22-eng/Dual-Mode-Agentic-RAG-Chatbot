import pytest
from backend.app.vector.retriever import BM25Retriever, HybridRetriever

def test_bm25_retriever():
    corpus = [
        {"id": "doc1", "content": "Our sick leave policy allows 12 days per year.", "metadata": {}},
        {"id": "doc2", "content": "Full-time employees are entitled to 18 days of paid annual leave.", "metadata": {}},
        {"id": "doc3", "content": "Northwind Gadgets observes 10 public holidays each year.", "metadata": {}}
    ]
    retriever = BM25Retriever(corpus)
    
    # Keyword search "sick leave"
    results = retriever.score("sick leave")
    assert len(results) == 3
    assert results[0][0]["id"] == "doc1"
    assert results[0][1] > 0.0
    
    # Keyword search "holidays"
    results = retriever.score("holidays")
    assert results[0][0]["id"] == "doc3"
    assert results[0][1] > 0.0

def test_hybrid_retriever_rrf_scoring():
    retriever = HybridRetriever()
    
    # Perform similarity search with RRF over populated ChromaDB
    results = retriever.similarity_search_rrf("What is the annual leave allowance?", top_k=3)
    
    assert len(results) > 0
    assert "rrf_score" in results[0]
    assert "content" in results[0]
    assert "metadata" in results[0]

def test_reranker_and_citations():
    retriever = HybridRetriever()
    
    # Verify retrieval with top_k and citations
    results = retriever.retrieve("What is the annual leave allowance?", top_k=2, use_hybrid=True)
    
    assert len(results) > 0
    assert len(results) <= 2
    assert "citation" in results[0]
    assert "source" in results[0]["metadata"]
    assert "page" in results[0]["metadata"]
    if retriever.reranker_model:
        assert "rerank_score" in results[0]
