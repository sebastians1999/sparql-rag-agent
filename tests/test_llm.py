"""
Test script for the Google Generative AI LLM using configuration from config.py
"""

import os
import sys
from dotenv import load_dotenv

# Add the parent directory to sys.path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load environment variables from .env file
load_dotenv()

from scr.agent.utils.config import LLMConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from scr.agent.utils.llm_utils import get_llm
from scr.agent.utils.config import Configuration

def test_google_llm():
    """Test the Google Generative AI LLM with the configuration from config.py"""
    
    # Get the LLM configuration from the config file
    config = Configuration()
    
    # Override provider to use Google for this test
    config.llm_config.provider = "google-genai"
    
    print(f"Testing Google Generative AI with the following configuration:")
    print(f"Provider: {config.llm_config.provider}")
    print(f"Model 1: {config.llm_config.model_1}")
    print(f"Temperature: {config.llm_config.temperature}")
    print(f"Max tokens: {config.llm_config.max_tokens}")
    
    # Check if API key is set
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        # Try to get it from config as fallback
        api_key = config.llm_config.api_key
        
    print(f"API key: {'*****' + api_key[-4:] if api_key else None}")
    if not api_key:
        print("Error: GOOGLE_API_KEY not found in environment variables or config")
        print("Please set the GOOGLE_API_KEY environment variable in your .env file")
        return
    
    # Set the API key in the config
    config.llm_config.api_key = api_key
    
    try:
        # Test with model_1 (default)
        print("\n--- Testing with model_1 (default) ---")
        
        # Use the get_llm function which now properly handles Google authentication
        llm = get_llm(config)
        
        # Test with a simple prompt
        prompt = "What is SPARQL and how is it used?"
        print(f"\nPrompt: {prompt}")
        response = llm.invoke(prompt)
        print(response)
        
    except Exception as e:
        print(f"\nError occurred while testing Google Generative AI: {str(e)}")
        raise  # Re-raise the exception to see the full traceback

if __name__ == "__main__":
    test_google_llm()
