"""
SPARQL Entity Indexing Pipeline

This module provides functionality to extract RDF classes from SPARQL endpoints,
process them, and index them in a vector database for semantic search.
"""

import os
import time
import logging
import argparse
from typing import Dict, Any, List, Optional
import pandas as pd
from dotenv import load_dotenv

from SPARQLWrapper import SPARQLWrapper, JSON
import certifi
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.schema import Document

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class SPARQLEntityIndexer:
    """Main class for SPARQL entity indexing pipeline."""
    
    def __init__(
        self,
        endpoint_url: str,
        collection_name: str,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6334,
        dense_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        sparse_model: str = "prithivida/Splade_PP_en_v1",
        batch_size: int = 100,
    ):
        """
        Initialize the SPARQL entity indexer.
        
        Args:
            endpoint_url: URL of the SPARQL endpoint
            collection_name: Name of the Qdrant collection
            qdrant_host: Host of the Qdrant server
            qdrant_port: Port of the Qdrant server
            dense_model: Name of the dense embedding model
            sparse_model: Name of the sparse embedding model
            batch_size: Batch size for adding documents to Qdrant
        """
        self.endpoint_url = endpoint_url
        self.collection_name = collection_name
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.dense_model = dense_model
        self.sparse_model = sparse_model
        self.batch_size = batch_size
        
        # Set SSL certificate file
        os.environ['SSL_CERT_FILE'] = certifi.where()
        
        # Initialize SPARQL client
        self.sparql = SPARQLWrapper(endpoint_url)
        self.sparql.setReturnFormat(JSON)
        
        # Initialize Qdrant client
        self.client = None
        self.vectordb = None
    
    def setup_vector_store(self, recreate: bool = False) -> None:
        """
        Set up the vector store.
        
        Args:
            recreate: Whether to recreate the collection if it exists
        """
        logger.info(f"Setting up vector store '{self.collection_name}'")
        
        self.client = QdrantClient(
            host=self.qdrant_host,
            grpc_port=self.qdrant_port,
            prefer_grpc=True
        )
        
        # Set the embedding models
        self.client.set_model(self.dense_model)
        self.client.set_sparse_model(self.sparse_model)
        
        # Check if collection exists
        if self.client.collection_exists(self.collection_name):
            if recreate:
                logger.info(f"Deleting existing collection '{self.collection_name}'")
                self.client.delete_collection(self.collection_name)
            else:
                logger.info(f"Collection '{self.collection_name}' already exists")
                self._setup_vectordb()
                return
        
        # Create collection
        logger.info(f"Creating collection '{self.collection_name}'")
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=self.client.get_fastembed_vector_params(),
            sparse_vectors_config=self.client.get_fastembed_sparse_vector_params(),
        )
        
        self._setup_vectordb()
        
        # Log collection info
        collection_info = self.client.get_collection(collection_name=self.collection_name)
        logger.info(f"Available vectors: {collection_info.config.params.vectors.keys()}")
    
    def _setup_vectordb(self) -> None:
        """Set up the vector database interface."""
        self.vectordb = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=FastEmbedEmbeddings(model_name=self.dense_model),
            sparse_embedding=FastEmbedSparse(model_name=self.sparse_model),
            vector_name=self.dense_model.replace("/", "-").lower(),
            sparse_vector_name=f"fast-sparse-{self.sparse_model.replace('/', '-').lower()}",
            retrieval_mode=RetrievalMode.HYBRID,
        )
    
    def extract_classes(self, query: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract RDF classes from the SPARQL endpoint.
        
        Args:
            query: Custom SPARQL query to execute. If None, a default query will be used.
            
        Returns:
            Dict containing the SPARQL results
        """
        if query is None:
            query = """
            SELECT DISTINCT ?class ?label ?comment
            WHERE {
              ?class a rdfs:Class .
              OPTIONAL { ?class rdfs:label ?label }
              OPTIONAL { ?class rdfs:comment ?comment }
            }
            ORDER BY ?class
            """
        
        logger.info(f"Executing SPARQL query on endpoint {self.endpoint_url}")
        self.sparql.setQuery(query)
        results = self.sparql.query().convert()
        
        num_results = len(results["results"]["bindings"])
        logger.info(f"Found {num_results} classes")
        
        return results
    
    def process_results(self, results: Dict[str, Any], filter_empty_labels: bool = True) -> List[Document]:
        """
        Process SPARQL results into document format.
        
        Args:
            results: SPARQL query results
            filter_empty_labels: Whether to filter out entries without labels
            
        Returns:
            List of processed documents
        """
        logger.info("Processing SPARQL results")
        
        # Create DataFrame for easier processing
        df = pd.DataFrame(results["results"]["bindings"])
        
        if filter_empty_labels:
            # Count before filtering
            total_count = len(df)
            
            # Filter out rows where label is missing
            df = df[df['label'].notna()]
            
            filtered_count = len(df)
            logger.info(f"Filtered {total_count - filtered_count} entries without labels. Kept {filtered_count} entries.")
        
        documents = []
        
        for _, row in df.iterrows():
            uri = row.get('class', {}).get('value', '') if row.get('class') else ''
            label = row.get('label', {}).get('value', '') if row.get('label') else ''
            comment = row.get('comment', {}).get('value', '') if row.get('comment') else ''
            
            # Create a combined text representation
            parts = []
            if label:
                parts.append(f"Label: {label}")
            if comment:
                parts.append(f"Description: {comment}")
                
            # Always include the URI
            uri_name = uri.split('/')[-1] if '/' in uri else uri
            parts.append(f"URI: {uri_name}")
            
            content = " ".join(parts)
            
            doc = Document(
                page_content=content,
                metadata={
                    "uri": uri,
                    "original_label": label,
                    "original_comment": comment,
                    "type": "class"
                }
            )
            documents.append(doc)
        
        logger.info(f"Converted {len(documents)} classes to documents")
        return documents
    
    def index_documents(self, documents: List[Document]) -> None:
        """
        Index documents in the vector store.
        
        Args:
            documents: List of documents to index
        """
        if not self.vectordb:
            logger.error("Vector store not set up. Call setup_vector_store() first.")
            return
        
        logger.info(f"Indexing {len(documents)} documents")
        
        start_time = time.time()
        self.vectordb.add_documents(documents, batch_size=self.batch_size)
        duration = time.time() - start_time
        
        logger.info(f"Indexed {len(documents)} documents in {duration:.2f} seconds")
    
    def run_pipeline(self, query: Optional[str] = None, filter_empty_labels: bool = True, recreate_collection: bool = False) -> None:
        """
        Run the full indexing pipeline.
        
        Args:
            query: Custom SPARQL query. If None, a default query will be used.
            filter_empty_labels: Whether to filter out entries without labels
            recreate_collection: Whether to recreate the vector store collection
        """
        logger.info("Starting SPARQL entity indexing pipeline")
        
        # Setup vector store
        self.setup_vector_store(recreate=recreate_collection)
        
        # Extract classes
        results = self.extract_classes(query=query)
        
        # Process results
        documents = self.process_results(results, filter_empty_labels=filter_empty_labels)
        
        # Index documents
        self.index_documents(documents)
        
        logger.info("Pipeline completed successfully")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="SPARQL Entity Indexing Pipeline")
    parser.add_argument("--endpoint", "-e", required=True, help="SPARQL endpoint URL")
    parser.add_argument("--collection", "-c", required=True, help="Qdrant collection name")
    parser.add_argument("--host", default="localhost", help="Qdrant host")
    parser.add_argument("--port", type=int, default=6334, help="Qdrant port")
    parser.add_argument("--dense-model", default="sentence-transformers/all-MiniLM-L6-v2", help="Dense embedding model")
    parser.add_argument("--sparse-model", default="prithivida/Splade_PP_en_v1", help="Sparse embedding model")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for indexing")
    parser.add_argument("--filter-empty", action="store_true", help="Filter out entries without labels")
    parser.add_argument("--recreate", action="store_true", help="Recreate collection if it exists")
    parser.add_argument("--query-file", help="File containing the SPARQL query")
    
    args = parser.parse_args()
    
    # Load query from file if provided
    query = None
    if args.query_file:
        with open(args.query_file, 'r') as f:
            query = f.read()
    
    # Initialize and run the pipeline
    indexer = SPARQLEntityIndexer(
        endpoint_url=args.endpoint,
        collection_name=args.collection,
        qdrant_host=args.host,
        qdrant_port=args.port,
        dense_model=args.dense_model,
        sparse_model=args.sparse_model,
        batch_size=args.batch_size
    )
    
    indexer.run_pipeline(
        query=query,
        filter_empty_labels=args.filter_empty,
        recreate_collection=args.recreate
    )


if __name__ == "__main__":
    main()