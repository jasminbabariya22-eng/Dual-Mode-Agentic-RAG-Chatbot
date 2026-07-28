import math
import re
import time
import threading
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import CrossEncoder
from backend.app.config import settings
from backend.app.core.logger import logger
from backend.app.vector.store import VectorStoreManager

class BM25Retriever:
    """Pure Python implementation of BM25 retrieval for sparse keyword search."""
    
    def __init__(self, corpus: List[Dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus  # List of dicts with {"id": ..., "content": ..., "metadata": ...}
        self.doc_len: List[int] = []
        self.avg_doc_len: float = 0.0
        self.doc_term_freqs: List[Dict[str, int]] = []
        self.doc_frequencies: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self._initialize()

    def _tokenize(self, text: str) -> List[str]:
        """Lowers case, removes non-alphanumeric chars, and tokenizes."""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        return [w for w in text.split() if w]

    def _initialize(self):
        """Initializes frequencies, term statistics, and computes IDF values."""
        num_docs = len(self.corpus)
        total_len = 0
        
        for doc in self.corpus:
            tokens = self._tokenize(doc["content"])
            total_len += len(tokens)
            self.doc_len.append(len(tokens))
            
            # Compute term frequencies for this document
            freqs: Dict[str, int] = {}
            for token in tokens:
                freqs[token] = freqs.get(token, 0) + 1
            self.doc_term_freqs.append(freqs)
            
            # Update overall document frequencies
            for token in freqs.keys():
                self.doc_frequencies[token] = self.doc_frequencies.get(token, 0) + 1
                
        self.avg_doc_len = total_len / num_docs if num_docs > 0 else 0.0
        
        # Compute IDF values
        for token, freq in self.doc_frequencies.items():
            # Standard BM25 IDF formulation
            self.idf[token] = math.log((num_docs - freq + 0.5) / (freq + 0.5) + 1.0)

    def score(self, query: str) -> List[Tuple[Dict[str, Any], float]]:
        """Scores all documents against query and returns scored list."""
        query_tokens = self._tokenize(query)
        scores: List[Tuple[Dict[str, Any], float]] = []
        
        for idx, doc in enumerate(self.corpus):
            score = 0.0
            doc_len = self.doc_len[idx]
            freqs = self.doc_term_freqs[idx]
            
            for token in query_tokens:
                if token in freqs:
                    tf = freqs[token]
                    idf = self.idf.get(token, 0.0)
                    # BM25 numerator & denominator
                    numerator = tf * (self.k1 + 1.0)
                    denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len if self.avg_doc_len > 0 else 1.0))
                    score += idf * (numerator / denominator)
            scores.append((doc, score))
            
        return sorted(scores, key=lambda x: x[1], reverse=True)


class CitationFormatter:
    """Handles parsing and formatting citations for responses."""
    
    @staticmethod
    def format(source: str, page: int) -> str:
        return f"[{source} (Page {page})]"


class HybridRetriever:
    """Orchestrates Dense (ChromaDB) and Sparse (BM25) search, fuses rankings with RRF, and reranks."""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(HybridRetriever, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
        
    def __init__(self):
        if getattr(self, "_initialized", False):
            return
            
        self.vstore = VectorStoreManager()
        self.reranker_model: Optional[CrossEncoder] = None
        self._load_reranker()
        self._initialized = True

    def _load_reranker(self):
        """Loads cross-encoder model for semantic reranking."""
        try:
            logger.info(f"Loading Reranker model: {settings.RERANKER_MODEL} on CPU...")
            self.reranker_model = CrossEncoder(settings.RERANKER_MODEL, device="cpu")
            logger.info("Reranker model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Reranker model: {str(e)}. Proceeding without reranking.")

    def _get_full_corpus(self) -> List[Dict[str, Any]]:
        """Loads all documents from ChromaDB collection to construct the BM25 index."""
        collection = self.vstore.get_collection()
        results = collection.get()
        
        corpus = []
        if results and "documents" in results:
            docs = results["documents"]
            metas = results["metadatas"]
            ids = results["ids"]
            for i in range(len(docs)):
                corpus.append({
                    "id": ids[i],
                    "content": docs[i],
                    "metadata": metas[i]
                })
        return corpus

    def similarity_search_rrf(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """Executes Dense and BM25 retrievals, fusses rankings via Reciprocal Rank Fusion (RRF)."""
        start_time = time.perf_counter()
        
        # Load corpus and build BM25 retriever
        corpus = self._get_full_corpus()
        if not corpus:
            logger.warning("Empty ChromaDB corpus. Cannot perform hybrid search.")
            return []
            
        # 1. Sparse BM25 Retrieval
        bm25 = BM25Retriever(corpus)
        bm25_results = bm25.score(query)
        
        # 2. Dense Vector Retrieval
        dense_results = self.vstore.similarity_search(query, top_k=len(corpus))
        
        # 3. Reciprocal Rank Fusion
        # Create rank maps
        dense_rank = {doc["id"]: idx for idx, doc in enumerate(dense_results)}
        bm25_rank = {doc[0]["id"]: idx for idx, doc in enumerate(bm25_results)}
        
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Dict[str, Any]] = {}
        
        # Document lookup map
        for doc in dense_results:
            doc_map[doc["id"]] = doc
        for doc, _ in bm25_results:
            if doc["id"] not in doc_map:
                # Add missing fields
                doc_map[doc["id"]] = {
                    "id": doc["id"],
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                    "score": 0.0
                }
                
        # Calculate RRF Scores
        # RRF score = 1 / (60 + rank_dense) + 1 / (60 + rank_sparse)
        constant_k = 60.0
        
        for doc_id, doc in doc_map.items():
            rrf_score = 0.0
            if doc_id in dense_rank:
                rrf_score += 1.0 / (constant_k + dense_rank[doc_id] + 1)
            if doc_id in bm25_rank:
                rrf_score += 1.0 / (constant_k + bm25_rank[doc_id] + 1)
            rrf_scores[doc_id] = rrf_score
            
        # Format and sort by RRF score
        fused_results = []
        for doc_id, score in rrf_scores.items():
            doc = doc_map[doc_id]
            doc["rrf_score"] = score
            fused_results.append(doc)
            
        fused_results = sorted(fused_results, key=lambda x: x["rrf_score"], reverse=True)
        
        latency = (time.perf_counter() - start_time) * 1000
        logger.info(f"[Retrieval] Dense + BM25 RRF fusion executed in {latency:.2f}ms. Returned {len(fused_results)} candidates.")
        
        return fused_results[:top_k]

    def retrieve(self, query: str, top_k: Optional[int] = None, use_hybrid: Optional[bool] = None) -> List[Dict[str, Any]]:
        """Performs retrieval (hybrid or dense), and executes Cross-Encoder Reranking."""
        from backend.app.core.cache import cache_manager
        
        if top_k is None:
            top_k = settings.FINAL_TOP_K
        if use_hybrid is None:
            use_hybrid = settings.ENABLE_HYBRID_SEARCH
            
        start_time = time.perf_counter()
        
        # Check standard cache
        cache_key = f"query:{query}:hybrid:{use_hybrid}"
        cached_val = cache_manager.get_kv("retriever", cache_key)
        if cached_val:
            import json
            latency = (time.perf_counter() - start_time) * 1000
            logger.info(f"[Retrieval Engine] Cache HIT in {latency:.2f}ms")
            return json.loads(cached_val)
            
        # Get candidates (Top Vector K from settings)
        candidates_k = settings.VECTOR_TOP_K
        if use_hybrid:
            candidates = self.similarity_search_rrf(query, top_k=candidates_k)
        else:
            candidates = self.vstore.similarity_search(query, top_k=candidates_k)
            
        if not candidates:
            return []
            
        # Rerank candidates if Reranker model is loaded and enabled
        if self.reranker_model and settings.ENABLE_RERANKER:
            start_rerank = time.perf_counter()
            pairs = [[query, doc["content"]] for doc in candidates]
            scores = self.reranker_model.predict(pairs)
            
            for idx, score in enumerate(scores):
                candidates[idx]["rerank_score"] = float(score)
                
            # Re-sort candidates based on reranker score
            candidates = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
            rerank_latency = (time.perf_counter() - start_rerank) * 1000
            logger.info(f"[Reranking] Reranked {len(candidates)} candidates in {rerank_latency:.2f}ms using {settings.RERANKER_MODEL}.")
        else:
            logger.info(f"[Reranking] Reranking skipped. Reranker enabled flag: {settings.ENABLE_RERANKER}")
            
        # Select top results (top_k parameter or configuration settings)
        final_results = candidates[:top_k]
        
        # Append citation formats
        for doc in final_results:
            meta = doc["metadata"]
            doc["citation"] = CitationFormatter.format(meta.get("source", "Unknown"), meta.get("page", 1))
            
        # Save to standard cache
        import json
        cache_manager.set_kv("retriever", cache_key, json.dumps(final_results), ttl=settings.CACHE_TTL)
        
        total_latency = (time.perf_counter() - start_time) * 1000
        logger.info(f"[Retrieval Engine] Search completed in {total_latency:.2f}ms. Top matches: {[d['id'] for d in final_results]}")
        
        return final_results
