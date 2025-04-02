from langchain_core.runnables import RunnableConfig
from scr.agent.state.state import State, StepOutput
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.documents import Document
from typing import List, Dict, Any
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from scr.agent.utils.config import Configuration
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage
from qdrant_client import QdrantClient
import os
import pathlib
import logging
import traceback

# Set up logging
logger = logging.getLogger(__name__)

def format_extracted_entities(entities_list):
    """Format the extracted entities for display in the agent UI."""
    result = ""
    for entity_data in entities_list:
        result += f"**Entity: {entity_data['text']}**\n\n"
        if entity_data['matchs']:
            result += "Matches:\n"
            for i, doc in enumerate(entity_data['matchs']):
                score = doc.metadata.get("score", 0)
                result += f"- [{score:.3f}] {doc.page_content}\n"
                result += f"  URI: {doc.metadata.get('uri', 'N/A')}\n"
                result += f"  Type: {doc.metadata.get('type', 'N/A')}\n\n"
        else:
            result += "No matches found.\n\n"
    return result

async def retrieve_documents(state: State, config: RunnableConfig) -> Dict[str, List[AIMessage]]:
    """
    Retrieve documents for entity resolution using dense vector search.
    
    Args:
        state: The current state object containing structured question with extracted entities
        config: Configuration for the runnable
        
    Returns:
        Dictionary with extracted entities and their matches
    """
    configuration = Configuration.from_runnable_config(config)
    
    # Get settings from the configuration
    rag_config = configuration.rag_config
    
    # Initialize Qdrant client using configuration
    vectordb_url = rag_config.vectordb_url
    collection_name = rag_config.collection_name
    top_k = rag_config.top_k
    
    # Ensure the embeddings cache directory exists
    cache_dir = rag_config.embeddings_cache_dir
    pathlib.Path(cache_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize the embeddings model for dense retrieval
        logger.info(f"Initializing FastEmbedEmbeddings with model {rag_config.dense_embedding_model}")
        embeddings = FastEmbedEmbeddings(
            model_name=rag_config.dense_embedding_model,
            cache_dir=cache_dir,
        )
        
        # Initialize the Qdrant client
        logger.info(f"Connecting to Qdrant at {vectordb_url}")
        client = QdrantClient(
            host=rag_config.host,
            grpc_port=rag_config.grpc_port,
            prefer_grpc=True,
            timeout=60
        )
        
        entities_list = []
        
        # Get all entities to process
        entities = state.structured_question.extracted_entities
        
        if not entities:
            logger.warning("No entities found in structured question")
            return {
                "extracted_entities": [],
                "steps": [
                    StepOutput(
                        label="No entities to link",
                        details="No entities were extracted from the question.",
                    )
                ],
            }
        
        # Generate embeddings for all entities in a single batch
        logger.info(f"Generating batch embeddings for {len(entities)} entities")
        entity_embeddings = embeddings.embed_documents(entities)
        
        # Process each entity with its embedding
        for i, potential_entity in enumerate(entities):
            try:
                dense_vector = entity_embeddings[i]
                
                if dense_vector is not None:
                    # Use direct query_points method as requested
                    logger.info(f"Querying Qdrant for entity: {potential_entity}")
                    results = client.query_points(
                        collection_name=collection_name,
                        query=dense_vector,
                        using="dense",
                        with_payload=True,
                        limit=top_k,
                    )
                    
                    # Process matches and remove duplicates
                    matches = []
                    for point in results.points:
                        # Convert Qdrant point to Document
                        doc = Document(
                            page_content=point.payload.get("label", ""),
                            metadata={
                                "uri": point.payload.get("uri", ""),
                                "type": point.payload.get("type", ""),
                                "description": point.payload.get("description", ""),
                                "score": point.score,
                                "endpoint_url": point.payload.get("endpoint_url", ""),
                            }
                        )
                            
                        # Check if this URI already exists in matches
                        is_duplicate = False
                        for m in matches: 
                            if m.metadata["uri"] == doc.metadata["uri"]:
                                is_duplicate = True
                                break
                            
                        if not is_duplicate:
                            matches.append(doc)
                    
                    logger.info(f"Found {len(matches)} unique matches for entity: {potential_entity}")
                    entities_list.append({
                        "matchs": matches,
                        "text": potential_entity,
                    })
                else:
                    # If embedding failed, add entity with no matches
                    logger.warning(f"Empty embedding vector for entity: {potential_entity}")
                    entities_list.append({
                        "matchs": [],
                        "text": potential_entity,
                    })
            except Exception as e:
                # Handle errors for individual entities
                logger.error(f"Error processing entity '{potential_entity}': {str(e)}")
                logger.debug(traceback.format_exc())
                entities_list.append({
                    "matchs": [],
                    "text": potential_entity,
                    "error": str(e)
                })
        
        return {
            "extracted_entities": entities_list,
            "steps": [
                StepOutput(
                    label=f"Linked {len(entities_list)} potential entities",
                    details=format_extracted_entities(entities_list),
                )
            ],
        }
    except Exception as e:
        # Handle errors during initialization
        error_message = f"Error in entity resolution: {str(e)}"
        logger.error(error_message)
        logger.debug(traceback.format_exc())
        
        # Return empty results with error information
        return {
            "extracted_entities": [{"text": entity, "matchs": [], "error": error_message} 
                                  for entity in state.structured_question.extracted_entities],
            "steps": [
                StepOutput(
                    label="Entity linking failed",
                    details=f"Error: {error_message}\n\nPlease check that the embedding model and Qdrant server are accessible.",
                )
            ],
        }
