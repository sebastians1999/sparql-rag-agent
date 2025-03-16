"""Utility functions for working with Language Models."""

from typing import Optional, Dict, Any

# Import base classes
from langchain_core.language_models import BaseChatModel

# Import Together model (default)
try:
    from langchain_together import Together
    TOGETHER_AVAILABLE = True
except ImportError:
    TOGETHER_AVAILABLE = False

# Import OpenAI model
try:
    from langchain_openai import ChatOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Import Anthropic model
try:
    from langchain_anthropic import ChatAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from scr.agent.utils.config import Configuration, LLMConfig


def get_llm(config: Optional[Configuration] = None) -> BaseChatModel:
    """
    Get an LLM instance based on the configuration.
    
    Args:
        config: The configuration object. If None, a default config will be created.
        
    Returns:
        An instance of a language model that implements the BaseChatModel interface.
        
    Raises:
        ValueError: If the provider is not supported or the required package is not installed.
    """
    if config is None:
        config = Configuration()
    
    llm_config = config.llm_config
    
    # Common parameters for all LLMs
    common_params = {
        "temperature": llm_config.temperature,
        "max_tokens": llm_config.max_tokens,
    }
    
    # Provider-specific instantiation
    if llm_config.provider.lower() == "together":
        if not TOGETHER_AVAILABLE:
            raise ImportError(
                "The 'langchain_together' package is not installed. "
                "Please install it with 'pip install langchain_together'."
            )
        return Together(
            model=llm_config.model_name,
            **common_params
        )
    
    elif llm_config.provider.lower() == "openai":
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "The 'langchain_openai' package is not installed. "
                "Please install it with 'pip install langchain_openai'."
            )
        return ChatOpenAI(
            model=llm_config.model_name,
            api_key=llm_config.api_key,
            **common_params
        )
    
    elif llm_config.provider.lower() == "anthropic":
        if not ANTHROPIC_AVAILABLE:
            raise ImportError(
                "The 'langchain_anthropic' package is not installed. "
                "Please install it with 'pip install langchain_anthropic'."
            )
        return ChatAnthropic(
            model=llm_config.model_name,
            api_key=llm_config.api_key,
            **common_params
        )
    
    # Add more providers as needed
    
    else:
        raise ValueError(f"Unsupported LLM provider: {llm_config.provider}")
