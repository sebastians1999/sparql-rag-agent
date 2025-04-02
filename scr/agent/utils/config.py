"""Define the configurable parameters for the SPARQL RAG agent."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Optional, Dict, Any, List

from langchain_core.runnables import RunnableConfig


@dataclass
class LLMConfig:
    """Configuration for the Language Model."""
    
    # provider: str = "together" 
    # model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"
    # model_name: str = "llama3-8b-8192"
    # provider: str = "groq"
    provider: str = "groq"
    model_name: str = "deepseek-r1-distill-llama-70b"
    temperature: float = 1.0
    max_tokens: Optional[int] = 1000
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    timeout: int = 60  # seconds
    api_key: Optional[str] = None
    api_base: Optional[str] = None



@dataclass
class RAGConfig:
    """Configuration for Retrieval Augmented Generation."""
    
    vectordb_url: str = "http://vectordb:6334/"
    host: str = "localhost"
    grpc_port: int = 6334

    collection_name: Optional[str] = "chebi_collection"
    dense_embedding_model: str = "BAAI/bge-small-en-v1.5"
    sparse_embedding_model: str = "Qdrant/bm25"
    embeddings_cache_dir: str = "./embeddings_model_cache"
    
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 5


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    
    log_level: str = "INFO"
    log_file: Optional[str] = None
    enable_console_logging: bool = True
    enable_file_logging: bool = False


@dataclass(kw_only=True)
class Configuration:
    """Main configuration for the agent runner."""
    
    llm_config: LLMConfig = field(default_factory=LLMConfig)
    rag_config: RAGConfig = field(default_factory=RAGConfig)
    logging_config: LoggingConfig = field(default_factory=LoggingConfig)
    
    recursion_limit: int = 10
    timeout: int = 300  # seconds
    
    # Additional parameters
    cache_results: bool = True
    cache_dir: str = ".cache"
    
    # Test mode flag for retrieval of Qdrant
    test_mode: bool = True
    
    
    # Custom parameters for specific nodes
    node_params: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # List of enabled nodes/components
    enabled_components: List[str] = field(default_factory=list)
    
    
    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> Configuration:
        """Create a RunnerConfig instance from a RunnableConfig object."""
        configurable = (config.get("configurable") or {}) if config else {}
        
        # Extract top-level fields
        _fields = {f.name for f in fields(cls) if f.init}
        top_level_params = {k: v for k, v in configurable.items() if k in _fields and k not in ["llm_config", "rag_config", "logging_config"]}
        
        # Extract nested config fields
        llm_config_params = configurable.get("llm_config", {})
        rag_config_params = configurable.get("rag_config", {})
        logging_config_params = configurable.get("logging_config", {})
        
        # Create nested configs
        llm_config = LLMConfig(**llm_config_params) if llm_config_params else LLMConfig()
        rag_config = RAGConfig(**rag_config_params) if rag_config_params else RAGConfig()
        logging_config = LoggingConfig(**logging_config_params) if logging_config_params else LoggingConfig()
        
        # Create and return the RunnerConfig
        return cls(
            llm_config=llm_config,
            rag_config=rag_config,
            logging_config=logging_config,
            **top_level_params
        )