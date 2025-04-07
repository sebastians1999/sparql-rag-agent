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
import gc
import json
import uuid
from upload_tracker import UploadTracker
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
class EmbeddingWorker: 
    def __init__(self, model_name_dense: str = None, model_name_sparse: str = None):

        self.dense_model = TextEmbedding(model_name=model_name_dense, cache_dir="./embeddings_model_cache")
        self.sparse_model = SparseTextEmbedding(model_name=model_name_sparse, cache_dir="./embeddings_model_cache")
        
    def encode_dense(self, texts: List[str]):
        return list(self.dense_model.embed(texts))
    
    def encode_sparse(self, texts: List[str]):
        return list(self.sparse_model.embed(texts))
    
    def cleanup(self):
        del self.dense_model
        del self.sparse_model
        gc.collect()


class EmbeddingPipeline:
    def __init__(
        self,
        collection_name: str = "biomedical_entities",
        dense_model_name: str = "BAAI/bge-small-en-v1.5",
        sparse_model_name: str = "prithivida/Splade_PP_en_v1",
        host: str = "localhost",
        grpc_port: int = 6334,
        retrieval_mode: str = RetrievalMode.HYBRID,
        num_workers: int = 4,
        log_dir: str = "./upload_logs",
        recreate_collection: bool = False
    ):
        self.dense_model_name = dense_model_name
        self.sparse_model_name = sparse_model_name
        self.recreate_collection = recreate_collection
        self.collection_name = collection_name
        self.retrieval_mode = retrieval_mode
        # Create a pool of workers once
        print("Creating embedding workers...")
        self.num_workers = num_workers
        self.worker_pool = [EmbeddingWorker.remote(model_name_dense=dense_model_name, model_name_sparse=sparse_model_name) for _ in range(self.num_workers)]
        
        # Initialize Qdrant client
        print("Initializing Qdrant client...")
        self.client = QdrantClient(
            host=host,
            grpc_port=grpc_port,
            prefer_grpc=True,
            timeout=60
        )

        self.client.set_model(self.dense_model_name)
        self.client.set_sparse_model(self.sparse_model_name)
    
        
        # Initialize upload tracker
        self.upload_tracker = UploadTracker(collection_name, log_dir)
        
        # Create collection if it doesn't exist
        self._initialize_collection()

    def _initialize_collection(self):
        """Initialize Qdrant collection with both dense and sparse vectors"""
        try:
            # Check if collection exists
            collections = self.client.get_collections()

            if self.recreate_collection:
                self.client.delete_collection(self.collection_name)
                print(f"Collection '{self.collection_name}' deleted successfully")
                print("Recreating collection...")
                
            if self.collection_name in [c.name for c in collections.collections] and not self.recreate_collection:
                print(f"Collection '{self.collection_name}' exists, skipping initialization")

            else:
                print(f"Creating new collection '{self.collection_name}'")

                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                    "dense": models.VectorParams(
                        size=384,
                        distance=models.Distance.COSINE
                    )
                },
                hnsw_config=models.HnswConfigDiff(
                    on_disk=True
                    #indexing_threshold=0
                ),  
                # sparse_vectors_config={
                #     "sparse": models.SparseVectorParams(
                #         index=SparseIndexParams(
                #             on_disk=True
                #         )
                #     )
                # }
                )
                print("Collection initialized successfully")
            
        except Exception as e:
            print(f"Error in collection initialization: {str(e)}")
            raise

    # def add_documents(self, documents: List[Dict[str, str]], batch_size: int = 144, skip_existing: bool = True):
    #     """Add documents to the collection in batches"""
        
    #     # Filter out already uploaded documents if skip_existing is True
    #     if skip_existing:
    #         documents = self.upload_tracker.filter_new_entities(documents)
    #         if not documents:
    #             print("All documents have already been uploaded. Skipping.")
    #             return
        
    #     total_docs = len(documents)
    #     print(f"Adding {total_docs} documents in batches of {batch_size}")
        
    #     batch_counter = 0
        
    #     for i in range(0, total_docs, batch_size):
    #         batch = documents[i:i + batch_size]
            
    #         try:
    #             # Generate embeddings for the batch
    #             texts = [doc['label'] for doc in batch]
    #             #print(texts)
    #             dense_embeddings = self.chunk_encode_dense(texts)
                
    #             #contains list of sparse embeddings objects. Each object has indices and values
    #             sparse_embeddings = self.chunk_encode_sparse(texts)
                
    #             gc.collect()
                
    #             points = []
    #             batch_ids = []
    #             print("Creating points...")
    #             for idx, (doc, dense_vector, sparse_vector) in enumerate(zip(batch, dense_embeddings, sparse_embeddings)):
    #                 # Convert sparse embedding to the format expected by Qdrant
    #                 sparse_vector = SparseVector(indices=sparse_vector.indices.tolist(), values=sparse_vector.values.tolist())
    #                 batch_ids.append(doc['uri'])

    #                 # Generate a UUID based on the URI for consistency
    #                 # This ensures the same URI always gets the same UUID
    #                 point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, doc['uri']))

    #                 point = PointStruct(
    #                     id=point_id,
    #                     vector={
    #                         "dense": dense_vector.tolist(),
    #                         "sparse": sparse_vector
    #                     },
    #                     payload={
    #                         "uri": doc['uri'],
    #                         "label": doc['label'],
    #                         "type": doc['entity_type'],
    #                         "description": doc.get('description', '')
    #                     }
    #                 )
    #                 #print(point.id)
    #                 points.append(point)

    #             print("Uploading points...")
    #             self.client.upload_points(
    #                 collection_name=self.collection_name,
    #                 points=points
    #             )
                
    #             # Mark batch as uploaded
    #             self.upload_tracker.mark_as_uploaded(batch_ids)
                
    #             print(f"✓ Batch {i//batch_size + 1}/{(total_docs + batch_size - 1)//batch_size} - Total uploaded: {self.upload_tracker.get_uploaded_count()}")
                
    #             batch_counter += 1

    #             if batch_counter == 100:
    #                 print("Shutting down Ray to free up resources...")
    #                 ray.shutdown()
    #                 print("Waiting 10 seconds...")
    #                 time.sleep(10)
    #                 batch_counter = 0
    #                 ray.init()
    #                 print("Initialized Ray again and recreated embedding workers...")
    #                 self.worker_pool = [EmbeddingWorker.remote(model_name_dense=self.dense_model_name, model_name_sparse=self.sparse_model_name) for _ in range(self.num_workers)]
                    
                
    #         except Exception as e:
    #             print(f"Error processing batch {i//batch_size + 1}: {str(e)}")
    #             continue
        
    #     # Enable indexing after all data is loaded
    #     # self.client.update_collection(
    #     #     collection_name=self.collection_name,
    #     #     optimizer_config=models.OptimizersConfigDiff(
    #     #         indexing_threshold=20000
    #     #     )
    #     # )
    #     # Cleanup resources
    #     # cleanup_tasks = [worker.cleanup.remote() for worker in self.worker_pool]
    #     # ray.get(cleanup_tasks)
    #     ray.shutdown()

    def chunk_encode_dense(self, texts: List[str]):
        # Use existing workers instead of creating new ones
        chunk_size = len(texts) // self.num_workers
        text_chunks = [texts[i:i+chunk_size] for i in range(0, len(texts), chunk_size)]
        print("Chunk size:", chunk_size)

        chunk_object_ref = [ray.put(text_chunk) for text_chunk in text_chunks]
        
        # Reuse worker pool which got initialized in __init__ of EmbeddingPipeline
        start_time = time.time()
        embedding_tasks_ref = [worker.encode_dense.remote(chunk_ref) for worker, chunk_ref in zip(self.worker_pool, chunk_object_ref)]
        embeddings = ray.get(embedding_tasks_ref)
        end_time = time.time()

        for refs in chunk_object_ref:
            del refs
        del chunk_object_ref
        gc.collect()

        for ref in embedding_tasks_ref:
            del ref
        del embedding_tasks_ref
        gc.collect()

        # Flatten the embeddings list
        embeddings = [embedding for sublist in embeddings for embedding in sublist]

        print("Time taken to generate dense embeddings with Ray Distributed Computing:", end_time - start_time, "seconds")

        #print("Generated embeddings:", embeddings)
        return embeddings



    def chunk_encode_sparse(self, texts: List[str]):
        # Use existing workers instead of creating new ones
        chunk_size = len(texts) // self.num_workers
        text_chunks = [texts[i:i+chunk_size] for i in range(0, len(texts), chunk_size)]

        chunk_object_ref = [ray.put(text_chunk) for text_chunk in text_chunks]

        # Reuse worker pool which got initialized in __init__ of EmbeddingPipeline
        start_time = time.time()
        embedding_tasks_ref = [worker.encode_sparse.remote(chunk_ref) for worker, chunk_ref in zip(self.worker_pool, chunk_object_ref)]
        embeddings = ray.get(embedding_tasks_ref)
        end_time = time.time()


        for refs in chunk_object_ref:
            del refs
        del chunk_object_ref
        gc.collect()

        for refs in embedding_tasks_ref:
            del refs
        del embedding_tasks_ref
        gc.collect()


        # Flatten the embeddings list
        embeddings = [embedding for sublist in embeddings for embedding in sublist]

        print("Time taken to generate sparse embeddings with Ray Distributed Computing:", end_time - start_time, "seconds")

        return embeddings

    def search(self,query_text: str):
        # Compute sparse and dense vectors
        dense_vector = self.dense_model.encode([query_text])[0]
        sparse_vector = self.sparse_model.encode([query_text])[0]
        
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




    def add_documents(self, documents: List[Dict[str, str]], batch_size: int = 144, skip_existing: bool = True):
        """Add documents to the collection in batches"""
        
        # Filter out already uploaded documents if skip_existing is True
        if skip_existing:
            documents = self.upload_tracker.filter_new_entities(documents)
            if not documents:
                print("All documents have already been uploaded. Skipping.")
                return
            
        total_docs = len(documents)
        print(f"Adding {total_docs} documents in batches of {batch_size}")
        
        batch_counter = 0
        
        for i in range(0, total_docs, batch_size):
            batch = documents[i:i + batch_size]
            
            try:
                # Generate embeddings for the batch
                texts = [doc['label'] for doc in batch]
                #print(texts)
                dense_embeddings = self.chunk_encode_dense(texts)
                    
                #contains list of sparse embeddings objects. Each object has indices and values
                #sparse_embeddings = self.chunk_encode_sparse(texts)
                    
                gc.collect()
                    
                points = []
                batch_uuids = []
                print("Creating points...")
                for idx, (doc, dense_vector) in enumerate(zip(batch, dense_embeddings)):
                    # Convert sparse embedding to the format expected by Qdrant
                    #sparse_vector = SparseVector(indices=sparse_vector.indices.tolist(), values=sparse_vector.values.tolist())
                    
                    # Generate a UUID based on the URI for consistency
                    # This ensures the same URI always gets the same UUID
                    uri = doc['uri']
                    point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, uri))
                    batch_uuids.append(point_id)

                    point = PointStruct(
                        id=point_id,
                        vector={
                            "dense": dense_vector.tolist(),
                                #"sparse": sparse_vector
                            },
                            payload={
                                "uri": uri,
                                "label": doc['label'],
                                "type": doc['entity_type'],
                                "description": doc.get('description', '')
                            }
                        )
                    #print(point.id)
                    points.append(point)

                print("Uploading points...")
                self.client.upload_points(
                    collection_name=self.collection_name,
                    points=points
                )
                    
                # Mark batch as uploaded using UUIDs instead of URIs
                self.upload_tracker.mark_as_uploaded(batch_uuids)
                    
                print(f"✓ Batch {i//batch_size + 1}/{(total_docs + batch_size - 1)//batch_size} - Total uploaded: {self.upload_tracker.get_uploaded_count()}")
                    
                batch_counter += 1

                if batch_counter == 10000:
                    print("Shutting down Ray to free up resources...")
                    ray.shutdown()
                    print("Waiting 10 seconds...")
                    time.sleep(10)
                    batch_counter = 0
                    ray.init()
                    print("Initialized Ray again and recreated embedding workers...")
                    self.worker_pool = [EmbeddingWorker.remote(model_name_dense=self.dense_model_name, model_name_sparse=self.sparse_model_name) for _ in range(self.num_workers)]
                        
            except Exception as e:
                print(f"Error processing batch {i//batch_size + 1}: {str(e)}")
                continue
            
            # Enable indexing after all data is loaded
            # self.client.update_collection(
            #     collection_name=self.collection_name,
            #     optimizer_config=models.OptimizersConfigDiff(
            #         indexing_threshold=20000
            #     )
            # )
            # Cleanup resources
            # cleanup_tasks = [worker.cleanup.remote() for worker in self.worker_pool]
            # ray.get(cleanup_tasks)
        
        ray.shutdown()
        
        print("All documents have been uploaded.")