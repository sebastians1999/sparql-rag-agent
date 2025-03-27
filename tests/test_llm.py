"""
Test script for the Groq LLM using configuration from config.py
"""

import os
import sys

# Add the parent directory to sys.path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scr.agent.utils.config import LLMConfig, Configuration
from langchain_groq import ChatGroq

def test_groq_llm():
    """Test the Groq LLM with the configuration from config.py"""
    
    # Get the LLM configuration from the config file
    llm_config = LLMConfig()
    
    # Override provider to use groq for this test
    llm_config.provider = "groq"
    
    print(f"Testing Groq LLM with the following configuration:")
    print(f"Provider: {llm_config.provider}")
    print(f"Model: {llm_config.model_name}")
    print(f"Temperature: {llm_config.temperature}")
    print(f"Max tokens: {llm_config.max_tokens}")
    
    # Check if API key is set
    api_key = llm_config.api_key or os.environ.get("GROQ_API_KEY")
    print(f"API key: {api_key}")
    if not api_key:
        print("Error: GROQ_API_KEY not found in environment variables or config")
        print("Please set the GROQ_API_KEY environment variable")
        return
    
    try:
        # Initialize the Groq LLM
        llm = ChatGroq(
            model=llm_config.model_name,
            temperature=llm_config.temperature,
            max_tokens=llm_config.max_tokens,
            groq_api_key=api_key
        )
        
        # Test with a simple prompt
        prompt = "What is the capital of France?"
        print(f"\nSending test prompt: '{prompt}'")
        
        response = llm.invoke(prompt)
        
        print("\nResponse received successfully:")
        print(response)
        
    except Exception as e:
        print(f"\nError occurred while testing Groq LLM: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_groq_llm()
