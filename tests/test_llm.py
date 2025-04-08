"""
Test script for different LLM providers and models using configuration from config.py
"""

import os
import sys
import time
from dotenv import load_dotenv

# Add the parent directory to sys.path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load environment variables from .env file
load_dotenv()

from scr.agent.utils.config import Configuration
from scr.agent.utils.llm_utils import get_llm

def test_provider(provider_key, model_key=None, prompt=None):
    """Test a specific LLM provider and model"""
    
    if prompt is None:
        prompt = "What is SPARQL and how is it used in semantic web applications? Keep your answer brief."
    
    # Get the configuration
    config = Configuration()
    
    # Get the provider name from the provider key
    provider_name = getattr(config.llm_config, provider_key)
    
    print(f"\n{'='*50}")
    print(f"Testing {provider_name} provider")
    print(f"{'='*50}")
    
    # Get the model name
    if model_key:
        model_attr = f"{provider_name}_{model_key}"
        if hasattr(config.llm_config, model_attr):
            model_name = getattr(config.llm_config, model_attr)
        else:
            print(f"Model key '{model_key}' not found for provider '{provider_name}'")
            return
    else:
        # Use default model for this provider
        model_attr = f"{provider_name}_model_1"
        if hasattr(config.llm_config, model_attr):
            model_name = getattr(config.llm_config, model_attr)
        else:
            print(f"No default model found for provider '{provider_name}'")
            return
    
    # Get the API key
    api_key_attr = f"{provider_name}_api_key"
    api_key = os.environ.get(f"{provider_name.upper().replace('-', '_')}_API_KEY")
    
    if not api_key and hasattr(config.llm_config, api_key_attr):
        api_key = getattr(config.llm_config, api_key_attr)
    
    # Display configuration
    print(f"Provider: {provider_name}")
    print(f"Model: {model_name}")
    print(f"API key: {'*****' + api_key[-4:] if api_key else 'Not found'}")
    print(f"Temperature: {config.llm_config.temperature}")
    print(f"Max tokens: {config.llm_config.max_tokens}")
    
    if not api_key:
        print(f"Error: API key for {provider_name} not found in environment variables or config")
        print(f"Please set the {provider_name.upper().replace('-', '_')}_API_KEY environment variable in your .env file")
        return
    
    # Set the API key in the config
    setattr(config.llm_config, api_key_attr, api_key)
    
    try:
        # Get the LLM
        start_time = time.time()
        print("\nInitializing LLM...")
        llm = get_llm(config, model_key=model_key, provider_key=provider_key)
        init_time = time.time() - start_time
        print(f"LLM initialized in {init_time:.2f} seconds")
        
        # Test with the prompt
        print(f"\nPrompt: {prompt}")
        start_time = time.time()
        response = llm.invoke(prompt)
        inference_time = time.time() - start_time
        
        print(f"\nResponse (generated in {inference_time:.2f} seconds):")
        print(response.content)
        
        return True
    except Exception as e:
        print(f"\nError occurred while testing {provider_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """Run tests for all available providers"""
    
    config = Configuration()
    
    # Get all provider keys
    provider_keys = [
        key for key in dir(config.llm_config) 
        if key.startswith("provider_") and not key.startswith("__")
    ]
    
    results = {}
    
    for provider_key in provider_keys:
        provider_name = getattr(config.llm_config, provider_key)
        result = test_provider(provider_key)
        results[provider_name] = result
    
    # Print summary
    print("\n" + "="*50)
    print("Test Results Summary")
    print("="*50)
    for provider, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{provider}: {status}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test LLM providers and models")
    parser.add_argument("--provider", type=str, help="Provider key (e.g., provider_1)")
    parser.add_argument("--model", type=str, help="Model key (e.g., model_1)")
    parser.add_argument("--prompt", type=str, help="Custom prompt to test with")
    parser.add_argument("--all", action="store_true", help="Test all available providers")
    
    args = parser.parse_args()
    
    if args.all:
        run_all_tests()
    elif args.provider:
        test_provider(args.provider, args.model, args.prompt)
    else:
        # Default: test the default provider
        config = Configuration()
        default_provider_key = config.llm_config.default_provider
        test_provider(default_provider_key)
