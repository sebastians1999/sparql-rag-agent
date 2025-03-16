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

# Test documents for development and testing
TEST_DOCUMENTS = [
    Document(
        page_content="TP53 (Tumor Protein P53) is a crucial tumor suppressor gene that plays a central role in preventing cancer formation. It encodes the p53 protein which regulates cell division and prevents tumor formation by controlling cell cycle and apoptosis.",
        metadata={
            "id": 1,
            "title": "TP53",
            "type": "gene",
            "organism": "Homo sapiens",
            "uniprot_id": "P04637",
            "function": "tumor suppressor"
        }
    ),
    Document(
        page_content="BRCA1 (Breast Cancer 1) is a human tumor suppressor gene that produces a protein called breast cancer type 1 susceptibility protein. It is responsible for repairing damaged DNA and maintaining genomic stability.",
        metadata={
            "id": 2,
            "title": "BRCA1",
            "type": "gene",
            "organism": "Homo sapiens",
            "uniprot_id": "P38398",
            "function": "DNA repair"
        }
    ),
    Document(
        page_content="EGFR (Epidermal Growth Factor Receptor) is a transmembrane protein that is a receptor for members of the epidermal growth factor family. It plays a crucial role in cell growth, survival, and differentiation.",
        metadata={
            "id": 3,
            "title": "EGFR",
            "type": "protein",
            "organism": "Homo sapiens",
            "uniprot_id": "P00533",
            "function": "receptor tyrosine kinase"
        }
    ),
    Document(
        page_content="KRAS (Kirsten Rat Sarcoma Viral Oncogene Homolog) is a gene that provides instructions for making a protein called K-Ras, which is part of the RAS/MAPK pathway. Mutations in this gene are commonly found in various types of cancer.",
        metadata={
            "id": 4,
            "title": "KRAS",
            "type": "gene",
            "organism": "Homo sapiens",
            "uniprot_id": "P01116",
            "function": "signal transduction"
        }
    ),
    Document(
        page_content="HER2 (Human Epidermal Growth Factor Receptor 2) is a protein that promotes the growth of cancer cells. It is overexpressed in about 20% of breast cancers and is a target for several cancer therapies.",
        metadata={
            "id": 5,
            "title": "HER2",
            "type": "protein",
            "organism": "Homo sapiens",
            "uniprot_id": "P04626",
            "function": "receptor tyrosine kinase"
        }
    )
]

async def retrieve_documents(state: State, config: RunnableConfig) -> Dict[str, List[AIMessage]]:
    # Test mode - return predefined documents

    configuration = Configuration.from_runnable_config(config)

    if configuration.test_mode:
        documents = TEST_DOCUMENTS
        entities_list = [doc.metadata["title"] for doc in documents]
        return {
            "retrieved_docs": documents,
            "steps": [
                StepOutput(
                    label=f"Linked {len(entities_list)} potential entities (Test Mode)",
                    details="Using predefined biomedical test documents for development and testing",
                )
            ],
        }

    # Production mode - use Qdrant vector DB
    # vectordb = QdrantVectorStore.from_existing_collection(
    #     # client=qdrant_client,
    #     url=configuration.rag_config.vectordb_url,
    #     prefer_grpc=True,
    #     collection_name=configuration.rag_config.collection_name,
    #     embedding=FastEmbedEmbeddings(model_name = configuration.rag_config.embedding_model),
    #     sparse_embedding=FastEmbedSparse(model_name=configuration.rag_config.sparse_embedding_model),
    #     retrieval_mode=RetrievalMode.HYBRID,
    # )

    # TODO: still need to implement the retrieval of the documents
    # TODO: need to add the documents to the state
    # TODO: Need a function that formats the documents to be used in the prompt
    # documents = [
    #     {"id": 1, "title": "Document 1", "content": "This is the content of document 1."},
    #     {"id": 2, "title": "Document 2", "content": "This is the content of document 2."},
    #     {"id": 3, "title": "Document 3", "content": "This is the content of document 3."},
    #     {"id": 4, "title": "Document 4", "content": "This is the content of document 4."},
    #     {"id": 5, "title": "Document 5", "content": "This is the content of document 5."},
    # ]
    # entities_list = [doc["title"] for doc in documents]  # Example of extracting titles as entities

    

    return {
        "extracted_entities": entities_list,
        "steps": [
            StepOutput(
                label=f"Linked {len(entities_list)} potential entities",
                details="This is a test",
            )
        ],
    }


  
