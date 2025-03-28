from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import SearchRequest, NamedVector, NamedSparseVector, SparseIndexParams, SparseVector
from langchain_qdrant.fastembed_sparse import FastEmbedSparse
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
import gc
import numpy as np
from fastembed.sparse import SparseTextEmbedding
import time
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


class FastEmbedDense_:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.dense_model = FastEmbedEmbeddings(
            model_name=model_name,
            parallel=4
        )
    
    def encode(self, texts: List[str]):
        return list(self.dense_model.embed_documents(texts))


class FastEmbedSparse_:
    def __init__(self, model_name: str = "prithivida/Splade_PP_en_v1"):
        self.sparse_model = FastEmbedSparse(
            model_name=model_name,
            parallel=4
        )
    
    def encode(self, texts: List[str]):
        return list(self.sparse_model.embed_documents(texts))


class EmbeddingPipeline:
    def __init__(
        self,
        collection_name: str = "biomedical_entities",
        dense_model_name: str = "BAAI/bge-small-en-v1.5",
        sparse_model_name: str = "prithivida/Splade_PP_en_v1",
        host: str = "localhost",
        grpc_port: int = 6334,
    ):
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
        self.retrieval_mode = RetrievalMode.HYBRID
        
        # Initialize embedders
        print("Initializing FastEmbed encoders")
        self.dense_encoder = FastEmbedDense_(model_name=dense_model_name)
        self.sparse_encoder = FastEmbedSparse_(model_name=sparse_model_name)
        
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
            
            sample_embedding = self.dense_encoder.encode(["test"])[0]
            vector_size = len(sample_embedding)

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": models.VectorParams(
                        size=vector_size,
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

    def add_documents(self, documents: List[Dict[str, str]], batch_size: int = 256):
        """Add documents to the collection in batches"""
        total_docs = len(documents)
        print(f"Adding {total_docs} documents in batches of {batch_size}")
        
        for i in range(0, total_docs, batch_size):
            batch = documents[i:i + batch_size]
            
            try:
                # Generate embeddings for the batch
                texts = [doc['label'] for doc in batch]


                start_time = time.time()
                dense_embeddings = self.dense_encoder.encode(texts)
                end_time = time.time()  
                print("Time taken to generate dense embeddings without Ray Distributed Computing:", end_time - start_time, "seconds")
                

                start_time = time.time()
                sparse_embeddings = self.sparse_encoder.encode(texts)
                end_time = time.time() 
                print("Time taken to generate sparse embeddings without Ray Distributed Computing:", end_time - start_time, "seconds")


                gc.collect()
                
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

    