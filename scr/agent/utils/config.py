from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List


@dataclass
class LLMConfig:
    """Configuration for the Language Model."""
    
    model_name: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
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
    collection_name: Optional[str] = "test_collection"
    # TODO: still need to change these models accordingly
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    sparse_embedding_model: str = "Qdrant/bm25"
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


@dataclass
class RunnerConfig:
    """Main configuration for the agent runner."""
    
    llm_config: LLMConfig = field(default_factory=LLMConfig)
    rag_config: RAGConfig = field(default_factory=RAGConfig)
    logging_config: LoggingConfig = field(default_factory=LoggingConfig)
    
    recursion_limit: int = 10
    timeout: int = 300  # seconds
    
    # Additional parameters
    cache_results: bool = True
    cache_dir: str = ".cache"
    
    # Custom parameters for specific nodes
    node_params: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # List of enabled nodes/components
    enabled_components: List[str] = field(default_factory=list)
    
    def get_node_param(self, node_name: str, param_name: str, default: Any = None) -> Any:
        """Get a parameter for a specific node."""
        if node_name in self.node_params and param_name in self.node_params[node_name]:
            return self.node_params[node_name][param_name]
        return default 