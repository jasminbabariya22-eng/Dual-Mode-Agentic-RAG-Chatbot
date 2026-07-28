import json
import time
import hashlib
from typing import Optional, Dict, Any, Tuple
import redis
from backend.app.config import settings
from backend.app.core.logger import logger

class CacheManager:
    """Orchestrates standard key-value caching (Redis/In-Memory) and Vector Semantic caching."""
    
    def __init__(self):
        self.redis_client = None
        self.use_redis = False
        self.local_kv: Dict[str, Tuple[str, float]] = {}  # key -> (value, expiry_timestamp)
        
        # Initialize Redis
        try:
            logger.info(f"Connecting to Redis at {settings.REDIS_URL}...")
            self.redis_client = redis.from_url(settings.REDIS_URL, socket_timeout=2.0)
            self.redis_client.ping()
            self.use_redis = True
            logger.info("Connected to Redis successfully.")
        except Exception as e:
            logger.warning(f"Redis connection failed: {str(e)}. Falling back to local memory KV cache.")
            
        # Initialize Vector Store for Semantic Caching
        try:
            from backend.app.vector.store import VectorStoreManager
            self.vstore = VectorStoreManager()
            self.cache_collection = self.vstore.client.get_or_create_collection(
                name="semantic_cache",
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("Semantic cache ChromaDB collection initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize semantic cache ChromaDB: {str(e)}")
            self.cache_collection = None

    # --- Standard Key-Value Cache ---
    
    def _get_cache_key(self, prefix: str, data: str) -> str:
        """Helper to generate a SHA256 hashed cache key."""
        hashed = hashlib.sha256(data.encode('utf-8')).hexdigest()
        return f"{prefix}:{hashed}"

    def get_kv(self, prefix: str, key_data: str) -> Optional[str]:
        """Gets value from standard cache. Measures performance latency."""
        start_time = time.perf_counter()
        key = self._get_cache_key(prefix, key_data)
        
        val = None
        if self.use_redis:
            try:
                raw = self.redis_client.get(key)
                if raw:
                    val = raw.decode('utf-8')
            except Exception as e:
                logger.error(f"Redis get failed: {str(e)}")
        
        # Local fallback if Redis failed/disabled
        if val is None:
            entry = self.local_kv.get(key)
            if entry:
                value, expiry = entry
                if expiry > time.perf_counter():
                    val = value
                else:
                    del self.local_kv[key]  # Clean expired
        
        latency = (time.perf_counter() - start_time) * 1000
        if val is not None:
            logger.debug(f"[Redis Lookup] Hit for key '{key}' in {latency:.2f}ms")
        else:
            logger.debug(f"[Redis Lookup] Miss for key '{key}' in {latency:.2f}ms")
        return val

    def set_kv(self, prefix: str, key_data: str, value: str, ttl: int = 3600):
        """Sets value in standard cache with TTL."""
        key = self._get_cache_key(prefix, key_data)
        if self.use_redis:
            try:
                self.redis_client.setex(key, ttl, value)
                return
            except Exception as e:
                logger.error(f"Redis set failed: {str(e)}")
        
        # Local fallback
        expiry = time.perf_counter() + ttl
        self.local_kv[key] = (value, expiry)

    # --- Semantic Cache ---

    def check_semantic_cache(self, query: str) -> Optional[str]:
        """Looks up similar queries in the semantic cache.
        Returns cached answer if similarity matches or exceeds threshold.
        """
        if not self.cache_collection or not settings.USE_SEMANTIC_CACHE:
            return None
            
        start_time = time.perf_counter()
        try:
            # Embed the query
            query_embedding = self.vstore.embeddings.embed_query(query)
            
            # Query the semantic cache collection
            results = self.cache_collection.query(
                query_embeddings=[query_embedding],
                n_results=1
            )
            
            if results and results["documents"] and len(results["documents"]) > 0:
                docs = results["documents"][0]
                metas = results["metadatas"][0]
                distances = results["distances"][0]
                
                if docs and len(docs) > 0:
                    distance = distances[0]
                    # Cosine distance to similarity conversion
                    similarity = max(0.0, min(1.0, 1.0 - distance))
                    
                    latency = (time.perf_counter() - start_time) * 1000
                    logger.info(f"[Semantic Cache] Best match similarity: {similarity:.4f} (Threshold: {settings.SEMANTIC_CACHE_THRESHOLD}) in {latency:.2f}ms")
                    
                    if similarity >= settings.SEMANTIC_CACHE_THRESHOLD:
                        logger.info(f"[Semantic Cache] HIT. Returning cached answer for query: '{query}'")
                        return metas[0].get("response")
                        
        except Exception as e:
            logger.error(f"Error checking semantic cache: {str(e)}")
            
        return None

    def store_semantic_cache(self, query: str, response: str):
        """Stores query, embedding, and response in semantic cache ChromaDB."""
        if not self.cache_collection or not settings.USE_SEMANTIC_CACHE:
            return
            
        try:
            query_embedding = self.vstore.embeddings.embed_query(query)
            doc_id = hashlib.sha256(query.encode('utf-8')).hexdigest()
            
            self.cache_collection.upsert(
                ids=[doc_id],
                embeddings=[query_embedding],
                documents=[query],
                metadatas=[{
                    "response": response,
                    "original_question": query,
                    "created_at": time.time()
                }]
            )
            logger.info(f"[Semantic Cache] Stored entry for: '{query}'")
        except Exception as e:
            logger.error(f"Error saving to semantic cache: {str(e)}")

    def clear_semantic_cache(self):
        """Clears all entries in the semantic cache collection."""
        if self.cache_collection:
            try:
                self.vstore.client.delete_collection("semantic_cache")
                self.cache_collection = self.vstore.client.get_or_create_collection(
                    name="semantic_cache",
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info("[Semantic Cache] Cleared successfully.")
            except Exception as e:
                logger.error(f"[Semantic Cache] Failed to clear: {str(e)}")

# Global instance for cache sharing
cache_manager = CacheManager()
