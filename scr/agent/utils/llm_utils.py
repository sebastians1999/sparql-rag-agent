"""Utility functions for working with Language Models."""

from typing import Optional, Dict, Any
import os

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

# Import Groq model
try:
    from langchain_groq import ChatGroq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# Import Google GenAI model
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GOOGLE_GENAI_AVAILABLE = True
except ImportError:
    GOOGLE_GENAI_AVAILABLE = False

from scr.agent.utils.config import Configuration, LLMConfig


def get_llm(config: Optional[Configuration] = None, model_key: Optional[str] = None) -> BaseChatModel:
    """
    Get an LLM instance based on the configuration.
    
    Args:
        config: The configuration object. If None, a default config will be created.
        model_key: The key of the model to use. If None, model_1 will be used.
                  Valid values are 'model_1', 'model_2', or any other model attribute defined in LLMConfig.
        
    Returns:
        An instance of a language model that implements the BaseChatModel interface.
        
    Raises:
        ValueError: If the provider is not supported or the required package is not installed.
    """
    if config is None:
        config = Configuration()
    
    llm_config = config.llm_config
    
    # Determine which model to use
    if model_key is not None:
        if hasattr(llm_config, model_key):
            model_name = getattr(llm_config, model_key)
        else:
            raise ValueError(f"Model key '{model_key}' not found in LLMConfig")
    else:
        # Default to model_1 if no model_key is specified
        model_name = llm_config.model_1
    
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
            model=model_name,
            **common_params
        )
    
    elif llm_config.provider.lower() == "openai":
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "The 'langchain_openai' package is not installed. "
                "Please install it with 'pip install langchain_openai'."
            )
        return ChatOpenAI(
            model=model_name,
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
            model=model_name,
            api_key=llm_config.api_key,
            **common_params
        )
    elif llm_config.provider.lower() == "groq":
        if not GROQ_AVAILABLE:
            raise ImportError(
                "The 'langchain_groq' package is not installed. "
                "Please install it with 'pip install langchain_groq'."
            )
        return ChatGroq(
            model=model_name,
            api_key=llm_config.api_key,
            **common_params
        )
    elif llm_config.provider.lower() == "google-genai":
        if not GOOGLE_GENAI_AVAILABLE:
            raise ImportError(
                "The 'langchain_google_genai' package is not installed. "
                "Please install it with 'pip install langchain_google_genai'."
            )
        
        
        return ChatGoogleGenerativeAI(
            model=model_name,
            api_key=llm_config.api_key,
            **common_params
        )
    
    else:
        raise ValueError(f"Unsupported LLM provider: {llm_config.provider}")
