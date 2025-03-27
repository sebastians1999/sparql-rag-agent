from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import SearchRequest, NamedVector, NamedSparseVector, SparseIndexParams, SparseVector
from fastembed import SparseEmbedding, TextEmbedding
import numpy as np
from fastembed.sparse import SparseTextEmbedding
import math
import os
import time
import ray
from qdrant_client.models import (
    Distance,
    NamedSparseVector,
    NamedVector,
    SparseVector,
    PointStruct,
    SearchRequest,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
    ScoredPoint,
)

class RetrievalMode:
    HYBRID = "hybrid"
    DENSE = "dense"
    SPARSE = "sparse"


@ray.remote
class EmbeddingWorkerDense: 
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):

        self.dense_model = TextEmbedding(model_name=model_name, cache_dir="./embeddings_model_cache")
        
    def encode(self, texts: List[str]):
        return list(self.dense_model.embed(texts))
    
    def cleanup(self):
        del self.dense_model
        import gc
        gc.collect()


class FastEmbedSparse:
    def __init__(self, model_name: str = "prithivida/Splade_PP_en_v1"):
        self.sparse_model = SparseTextEmbedding(
            model_name=model_name,
            batch_size=32
        )
    
    def encode(self, texts: List[str]):
        return list(self.sparse_model.embed(texts))


class EmbeddingPipeline:
    def __init__(
        self,
        collection_name: str = "biomedical_entities",
        dense_model_name: str = "BAAI/bge-small-en-v1.5",
        sparse_model_name: str = "prithivida/Splade_PP_en_v1",
        host: str = "localhost",
        grpc_port: int = 6334,
        retrieval_mode: str = RetrievalMode.HYBRID,
        num_workers: int = 4
        
    ):
        # Create a pool of workers once
        self.num_workers = num_workers
        self.worker_pool = [EmbeddingWorkerDense.remote(model_name=dense_model_name) for _ in range(self.num_workers)]
        
        # Initialize Qdrant client
        self.client = QdrantClient(
            host=host,
            grpc_port=grpc_port,
            prefer_grpc=True,
            timeout=60
        )

        self.client.set_model(dense_model_name)
        self.client.set_sparse_model(sparse_model_name)
        
        self.collection_name = collection_name
        self.retrieval_mode = retrieval_mode
        
        # Initialize embedders
        print("Initializing FastEmbed encoders")
        self.sparse_encoder = FastEmbedSparse(model_name=sparse_model_name)
        
        # Create collection if it doesn't exist
        self._initialize_collection()

    def _initialize_collection(self):
        """Initialize Qdrant collection with both dense and sparse vectors"""
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            if self.collection_name in [c.name for c in collections.collections]:
                print(f"Collection '{self.collection_name}' exists, recreating...")
                self.client.delete_collection(self.collection_name)
            
            print(f"Creating new collection '{self.collection_name}'")

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": models.VectorParams(
                        size=384,
                        distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(
                        index=SparseIndexParams(
                            on_disk=True
                        )
                    )
                }
            )
            print("Collection initialized successfully")
            
        except Exception as e:
            print(f"Error in collection initialization: {str(e)}")
            raise

    def add_documents(self, documents: List[Dict[str, str]], batch_size: int = 64):
        """Add documents to the collection in batches"""
        total_docs = len(documents)
        print(f"Adding {total_docs} documents in batches of {batch_size}")
        
        for i in range(0, total_docs, batch_size):
            batch = documents[i:i + batch_size]
            
            try:
                # Generate embeddings for the batch
                texts = [doc['label'] for doc in batch]
                print(texts)
                dense_embeddings = self.chunk_encode_dense(texts)
                #contains list of sparse embeddings objects. Each object has indices and values
                sparse_embeddings = self.sparse_encoder.encode(texts)
                
                points = []
                for idx, (doc, dense_vector, sparse_vector) in enumerate(zip(batch, dense_embeddings, sparse_embeddings)):
                    # Convert sparse embedding to the format expected by Qdrant
                    sparse_vector = SparseVector(indices=sparse_vector.indices.tolist(), values=sparse_vector.values.tolist())
                    point = PointStruct(
                        id=i + idx,
                        vector={
                            "dense": dense_vector.tolist(),
                            "sparse": sparse_vector
                        },
                        payload={
                            "uri": doc['uri'],
                            "label": doc['label'],
                            "type": doc.get('type', 'unknown'),
                            "description": doc.get('description', '')
                        }
                    )
                    points.append(point)
                
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                print(f"✓ Batch {i//batch_size + 1}/{(total_docs + batch_size - 1)//batch_size}")
                
            except Exception as e:
                print(f"Error processing batch {i//batch_size + 1}: {str(e)}")
                continue
        
        # Cleanup resources
        cleanup_tasks = [worker.cleanup.remote() for worker in self.worker_pool]
        ray.get(cleanup_tasks)
        ray.shutdown()


    def search(self,query_text: str):
        # Compute sparse and dense vectors
        dense_vector = self.dense_encoder.encode([query_text])[0]
        sparse_vector = self.sparse_encoder.encode([query_text])[0]
        
        # Convert sparse embedding to the format expected by Qdrant
        sparse_vector_qdrant = SparseVector(
            indices=sparse_vector.indices.tolist(),
            values=sparse_vector.values.tolist()
        )

        search_results = self.client.search_batch(
            collection_name=self.collection_name,
            requests=[
                SearchRequest(
                    vector=NamedVector(
                        name="dense",
                        vector=dense_vector.tolist(),
                    ),
                    limit=10,
                    with_payload=True,
                ),
                SearchRequest(
                    vector=NamedSparseVector(
                        name="sparse",
                        vector=sparse_vector_qdrant,
                    ),
                    limit=10,
                    with_payload=True,
                ),
            ],
        )

        return search_results
    
    def chunk_encode_dense(self, texts: List[str]):
        # Use existing workers instead of creating new ones
        chunk_size = max(1, len(texts) // self.num_workers)
        document_chunks = [texts[i:i+chunk_size] for i in range(0, len(texts), chunk_size)]
        
        # Reuse worker pool which got initialized in __init__ of EmbeddingPipeline
        start_time = time.time()
        embedding_tasks = [worker.encode.remote(chunk) for worker, chunk in zip(self.worker_pool, document_chunks)]
        embeddings = ray.get(embedding_tasks)
        end_time = time.time()

        # Flatten the embeddings list
        embeddings = [embedding for sublist in embeddings for embedding in sublist]

        print("Time taken to generate embeddings with Ray Distributed Computing:", end_time - start_time, "seconds")

        print("Generated embeddings:", embeddings)
        return embeddings
    