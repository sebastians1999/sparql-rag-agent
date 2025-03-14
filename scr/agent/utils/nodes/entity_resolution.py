from config import RunnerConfig
from state import State, StepOutput
from langchain_core.vectorstores import QdrantVectorStore
from langchain.embeddings import FastEmbedEmbeddings, FastEmbedSparse
from langchain_core.retrieval import RetrievalMode
from config import RunnerConfig
from state import State





async def retrieve_documents(state: State, config: RunnerConfig) -> dict[str,]:



    vectordb = QdrantVectorStore.from_existing_collection(
        # client=qdrant_client,
        url=config.rag_config.vectordb_url,
        prefer_grpc=True,
        collection_name=config.rag_config.collection_name,
        embedding=FastEmbedEmbeddings(model_name = config.rag_config.embedding_model),
        sparse_embedding=FastEmbedSparse(model_name=config.rag_config.sparse_embedding_model),
        retrieval_mode=RetrievalMode.HYBRID,
    )


    # TODO: still need to implement the retrieval of the documents
    # TODO: need to add the documents to the state
    # TODO: Need a function that formats the documents to be used in the prompt
    documents = [
        {"id": 1, "title": "Document 1", "content": "This is the content of document 1."},
        {"id": 2, "title": "Document 2", "content": "This is the content of document 2."},
        {"id": 3, "title": "Document 3", "content": "This is the content of document 3."},
        {"id": 4, "title": "Document 4", "content": "This is the content of document 4."},
        {"id": 5, "title": "Document 5", "content": "This is the content of document 5."},
    ]
    entities_list = [doc["title"] for doc in documents]  # Example of extracting titles as entities

    

    return {
        "extracted_entities": entities_list,
        "steps": [
            StepOutput(
                label=f"Linked {len(entities_list)} potential entities",
                details="This is a test",
            )
        ],
    }


  
