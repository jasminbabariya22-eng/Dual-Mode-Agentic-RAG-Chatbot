import pytest
import time
from backend.app.core.cache import cache_manager, CacheManager

def test_cache_fallback():
    cm = CacheManager()
    
    # Verify standard KV set and get (utilizes local dictionary if Redis is offline)
    key_data = "some_unique_query_parameters"
    prefix = "test"
    
    cm.set_kv(prefix, key_data, "cached_value", ttl=10)
    
    val = cm.get_kv(prefix, key_data)
    assert val == "cached_value"
    
    # Expiry check
    cm.set_kv(prefix, "expire_key", "will_expire", ttl=1)
    time.sleep(1.2)
    assert cm.get_kv(prefix, "expire_key") is None

def test_semantic_cache():
    cm = CacheManager()
    cm.clear_semantic_cache()
    
    # Test semantic queries
    query = "What is the warranty period for laptop stand?"
    answer = "The laptop stand is an accessory and has 6 months warranty."
    
    # Check cache (should miss first)
    res_miss = cm.check_semantic_cache(query)
    assert res_miss is None
    
    # Store entry
    cm.store_semantic_cache(query, answer)
    
    # Check cache with exact query (should HIT)
    res_hit = cm.check_semantic_cache(query)
    assert res_hit == answer
    
    # Check cache with semantically similar query (should HIT if similarity >= threshold)
    similar_query = "how long is laptop stand warranty?"
    res_sem_hit = cm.check_semantic_cache(similar_query)
    assert res_sem_hit == answer
