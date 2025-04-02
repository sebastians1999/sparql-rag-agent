from qdrant_client import QdrantClient
from ranx import Run, Qrels
from itertools import islice
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
from qdrant_client import QdrantClient, models



class Retriever:
    """
    A flexible retriever class for testing different retrieval methods.
    Supports dense, sparse, and hybrid retrieval from Qdrant.
    """
    
    def __init__(self, client: QdrantClient, collection_name: str):
        """
        Initialize the retriever.
        
        Args:
            client: QdrantClient instance
            collection_name: Name of the collection to query
        """
        self.client = client
        self.collection_name = collection_name
    
    def retrieve_dense(self, 
                      query_embeddings: Dict[str, Any], 
                      limit: int = 10, 
                      with_payload: bool = True) -> Dict[str, Dict[str, float]]:
        """
        Perform dense vector retrieval.
        
        Args:
            query_embeddings: Dictionary mapping query IDs to embedding dictionaries
            limit: Maximum number of results to return per query
            with_payload: Whether to include payload in results
            
        Returns:
            Dictionary mapping query IDs to dictionaries of (URI, score) pairs
        """
        run_dict = {}
        
        for idx, query in query_embeddings.items():
            dense_vector = query.get("dense_vector_query")
            
            if dense_vector is not None:
                results = self.client.query_points(
                    self.collection_name,
                    query=dense_vector,
                    using="dense",
                    with_payload=with_payload,
                    limit=limit,
                )
                
                run_dict[idx] = {
                    point.payload.get("uri", str(point.id)): point.score
                    for point in results.points
                }
            else:
                print(f"Warning: No dense vector found for query {idx}")
        
        return run_dict
    
    def retrieve_sparse(self, 
                       query_embeddings: Dict[str, Any], 
                       limit: int = 10, 
                       with_payload: bool = True) -> Dict[str, Dict[str, float]]:
        """
        Perform sparse vector retrieval.
        
        Args:
            query_embeddings: Dictionary mapping query IDs to embedding dictionaries
            limit: Maximum number of results to return per query
            with_payload: Whether to include payload in results
            
        Returns:
            Dictionary mapping query IDs to dictionaries of (URI, score) pairs
        """
        run_dict = {}
        
        for idx, query in query_embeddings.items():
            query_vector = query.get("sparse_vector_query")
            
            if query_vector is not None:
                qdrant_sparse_vector = models.SparseVector(
                    indices=query_vector.indices,
                    values=query_vector.values
                )
                
                results = self.client.query_points(
                    self.collection_name,
                    query=qdrant_sparse_vector,
                    using="sparse",
                    with_payload=with_payload,
                    limit=limit,
                )
                
                run_dict[idx] = {
                    point.payload.get("uri", str(point.id)): point.score
                    for point in results.points
                }
            else:
                print(f"Warning: No sparse vector found for query {idx}")
                
        return run_dict
    
    def retrieve_hybrid(self, 
                       query_embeddings: Dict[str, Any], 
                       limit: int = 10, 
                       with_payload: bool = True,
                       dense_weight: float = 0.5) -> Dict[str, Dict[str, float]]:
        """
        Perform hybrid (dense + sparse) retrieval.
        
        Args:
            query_embeddings: Dictionary mapping query IDs to embedding dictionaries
            limit: Maximum number of results to return per query
            with_payload: Whether to include payload in results
            dense_weight: Weight for dense scores (1-dense_weight will be used for sparse)
            
        Returns:
            Dictionary mapping query IDs to dictionaries of (URI, score) pairs
        """
        run_dict = {}

    
        
        for idx, query in query_embeddings.items():
            dense_vector = query.get("dense_vector_query")
            sparse_vector = query.get("sparse_vector_query")


            if dense_vector is not None and sparse_vector is not None:

                qdrant_sparse_vector = models.SparseVector(
                        indices=sparse_vector.indices,
                        values=sparse_vector.values
                    )

                prefetch = [models.Prefetch(
                                query=dense_vector,
                                using="dense",
                                limit=limit,
                            ),
                            models.Prefetch(
                                query=qdrant_sparse_vector,
                                using="sparse",
                                limit=limit,
                            ),]

                results = self.client.query_points(
                    self.collection_name,
                    prefetch=prefetch,
                    query=models.FusionQuery(
            fusion=models.Fusion.RRF,
        ),
                    with_payload=True,
                    limit=limit,
                )
                run_dict[idx] = {
                    point.payload.get("uri", str(point.id)): point.score
                    for point in results.points
                }
                # Print URIs outside the dictionary comprehension
                for point in results.points:
                    print(point.payload.get("uri", str(point.id)))
            else:
                print(f"Warning: Missing dense or sparse vector for query {idx}")
        
        return run_dict
    
    def batch_retrieve(self, 
                      query_embeddings: Dict[str, Any],
                      method: str = "dense",
                      limit: int = 10,
                      with_payload: bool = True,
                      num_elements: Optional[int] = None,
                      dense_weight: float = 0.5) -> Dict[str, Dict[str, float]]:
        """
        Batch retrieve using the specified method.
        
        Args:
            query_embeddings: Dictionary mapping query IDs to embedding dictionaries
            method: Retrieval method ("dense", "sparse", or "hybrid")
            limit: Maximum number of results to return per query
            with_payload: Whether to include payload in results
            num_elements: Optional limit on number of queries to process
            dense_weight: Weight for dense scores in hybrid retrieval
            
        Returns:
            Dictionary mapping query IDs to dictionaries of (URI, score) pairs
        """
        # Limit the number of queries if specified
        if num_elements is not None:
            query_embeddings = dict(islice(query_embeddings.items(), num_elements))
        
        if method == "dense":
            return self.retrieve_dense(query_embeddings, limit, with_payload)
        elif method == "sparse":
            return self.retrieve_sparse(query_embeddings, limit, with_payload)
        elif method == "hybrid":
            return self.retrieve_hybrid(query_embeddings, limit, with_payload, dense_weight)
        else:
            raise ValueError(f"Unknown retrieval method: {method}")
    
    def create_run(self, 
                  query_embeddings: Dict[str, Any],
                  method: str = "dense",
                  limit: int = 10,
                  with_payload: bool = True,
                  num_elements: Optional[int] = None,
                  dense_weight: float = 0.5,
                  name: Optional[str] = None) -> Run:
        """
        Create a ranx Run object for evaluation.
        
        Args:
            query_embeddings: Dictionary mapping query IDs to embedding dictionaries
            method: Retrieval method ("dense", "sparse", or "hybrid")
            limit: Maximum number of results to return per query
            with_payload: Whether to include payload in results
            num_elements: Optional limit on number of queries to process
            dense_weight: Weight for dense scores in hybrid retrieval
            name: Name for the Run object
            
        Returns:
            ranx Run object
        """
        run_dict = self.batch_retrieve(
            query_embeddings=query_embeddings,
            method=method,
            limit=limit,
            with_payload=with_payload,
            num_elements=num_elements,
            dense_weight=dense_weight
        )
        
        if name is None:
            name = method
        
        return Run(run_dict, name=name)
    
    @staticmethod
    def create_qrels(qrels_dict: Dict[str, Dict[str, int]]) -> Qrels:
        """
        Create a ranx Qrels object for evaluation.
        
        Args:
            qrels_dict: Dictionary mapping query IDs to dictionaries of (URI, relevance) pairs
            
        Returns:
            ranx Qrels object
        """
        return Qrels(qrels_dict)