import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from backend.app.config import settings
from backend.app.core.logger import logger

class VectorStoreManager:
    def __init__(self):
        self.chroma_path = settings.CHROMA_DB_PATH
        self.chroma_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize standard HF embeddings model
        logger.info(f"Initializing embedding model: {settings.EMBEDDINGS_MODEL}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDINGS_MODEL,
            model_kwargs={'device': 'cpu'},  # Default to CPU for portability
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize ChromaDB client (persistent on disk)
        self.client = chromadb.PersistentClient(
            path=str(self.chroma_path)
        )
        
        self.collection_name = "policies"
        self.collection = None
        
    def reset_collection(self):
        """Deletes the collection if it exists to allow changes in embedding dimensions, and immediately recreates it."""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted existing collection: {self.collection_name}")
        except Exception:
            pass
        # Recreate immediately to ensure self.collection points to a valid newly created collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def get_collection(self):
        """Gets or creates the ChromaDB collection with safe retrieval wrap."""
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
        except Exception:
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        return self.collection

    def add_documents(self, texts: List[str], metadatas: List[Dict[str, Any]], ids: List[str]):
        """Generates embeddings and adds text documents to the ChromaDB collection."""
        collection = self.get_collection()
        if collection is None:
            collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            self.collection = collection
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.embeddings.embed_documents(texts)
        
        # Upsert documents in ChromaDB
        collection.upsert(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        logger.info(f"Ingestion of {len(texts)} chunks to ChromaDB completed.")

    def similarity_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Queries the vector database for matches."""
        collection = self.get_collection()
        query_embedding = self.embeddings.embed_query(query)
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        formatted_results = []
        if results and results["documents"] and len(results["documents"]) > 0:
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            distances = results["distances"][0]
            ids = results["ids"][0]
            
            for i in range(len(docs)):
                # ChromaDB cosine distance range is [0, 2] where 0 is identical, 2 is opposite.
                # Cosine similarity = 1 - distance
                formatted_results.append({
                    "id": ids[i],
                    "content": docs[i],
                    "metadata": metas[i],
                    "distance": distances[i],
                    "score": max(0.0, min(1.0, 1.0 - distances[i]))
                })
        
        return formatted_results
