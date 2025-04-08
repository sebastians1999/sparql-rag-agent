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


def get_llm(config: Optional[Configuration] = None, model_key: Optional[str] = None, provider_key: Optional[str] = None) -> BaseChatModel:
    """
    Get an LLM instance based on the configuration.
    
    Args:
        config: The configuration object. If None, a default config will be created.
        model_key: The model name to use. If None, model_1 will be used.
        provider_key: The provider key to use. If None, the default provider will be used.
        
    Returns:
        An instance of a language model that implements the BaseChatModel interface.
        
    Raises:
        ValueError: If the provider is not supported or the required package is not installed.
    """
    if config is None:
        config = Configuration()
    
    llm_config = config.llm_config
    
    if provider_key is not None:
        provider_name = getattr(llm_config, provider_key)
    else:
        print("Error, no provider key specified or key not found in config")
        return None
    
    if model_key is not None:
        model_name = getattr(llm_config, model_key)
    else:
        print("Error, no model key specified or key not found in config")
        return None
    
    common_params = {
        "temperature": llm_config.temperature,
        "max_tokens": llm_config.max_tokens,
    }
    
    if provider_name.lower() == "together":
        if not TOGETHER_AVAILABLE:
            raise ImportError(
                "The 'langchain_together' package is not installed. "
                "Please install it with 'pip install langchain_together'."
            )
        return Together(
            model=model_name,
            api_key=llm_config.together_api_key,
            **common_params
        )
    
    elif provider_name.lower() == "openai":
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "The 'langchain_openai' package is not installed. "
                "Please install it with 'pip install langchain_openai'."
            )
        return ChatOpenAI(
            model=model_name,
            api_key=llm_config.openai_api_key,
            **common_params
        )
    
    elif provider_name.lower() == "anthropic":
        if not ANTHROPIC_AVAILABLE:
            raise ImportError(
                "The 'langchain_anthropic' package is not installed. "
                "Please install it with 'pip install langchain_anthropic'."
            )
        return ChatAnthropic(
            model=model_name,
            api_key=llm_config.anthropic_api_key,
            **common_params
        )
    elif provider_name.lower() == "groq":
        if not GROQ_AVAILABLE:
            raise ImportError(
                "The 'langchain_groq' package is not installed. "
                "Please install it with 'pip install langchain_groq'."
            )
        return ChatGroq(
            model=model_name,
            api_key=llm_config.groq_api_key,
            **common_params
        )
    elif provider_name.lower() == "google-genai":
        if not GOOGLE_GENAI_AVAILABLE:
            raise ImportError(
                "The 'langchain_google_genai' package is not installed. "
                "Please install it with 'pip install langchain_google_genai'."
            )
        return ChatGoogleGenerativeAI(
            model=model_name,
            api_key=llm_config.google_genai_api_key,
            **common_params
        )
    
    else:
        raise ValueError(f"Unsupported LLM provider: {provider_name}")
