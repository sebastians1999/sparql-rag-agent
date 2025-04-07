"""
Test script for structured output parsing with Google Generative AI
"""

import os
import sys
import asyncio
from dotenv import load_dotenv

# Add the parent directory to sys.path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load environment variables from .env file
load_dotenv()

from scr.agent.utils.config import Configuration
from scr.agent.state.state import StructuredQuestion
from scr.agent.utils.llm_utils import get_llm
from langchain_core.prompts import ChatPromptTemplate

# Sample JSON structure that should be parsed into StructuredQuestion
SAMPLE_JSON = """
{
  "intent": "general_information",
  "extracted_classes": ["Disease", "Symptom", "Alzheimer"],
  "extracted_entities": ["Alzheimer's disease", "symptoms"],
  "question_steps": [
    "Identify Alzheimer's disease in the knowledge base",
    "Find symptoms associated with Alzheimer's disease",
    "Return information about symptoms of Alzheimer's disease"
  ]
}
"""

async def test_structured_output_parsing():
    """Test the structured output parsing with Google Generative AI"""
    
    print("Testing structured output parsing with Google Generative AI...")
    
    # Create a test configuration
    config = Configuration()
    
    # Make sure we're using Google Generative AI
    config.llm_config.provider = "google-genai"
    config.llm_config.model_1 = "gemini-2.0-flash"
    
    # Check if API key is set
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not found in environment variables")
        return
    
    # Set the API key in the config
    config.llm_config.api_key = api_key
    
    try:
        # Get the LLM with structured output
        llm = get_llm(config).with_structured_output(schema=StructuredQuestion)
        
        # Create a simple prompt that asks the LLM to return the sample JSON structure
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that helps to answer questions about bioinformatics."),
            ("user", f"What are proteins in the human body that are associated with Alzheimer's disease and what are their functions?")
        ])
        
        # Format the messages
        messages = prompt_template.format_messages()
        
        print("\nSending request to the LLM...")
        
        # Invoke the LLM with the formatted messages
        response = await llm.ainvoke(messages)
        
        print("\nResponse received!")
        print(f"Response type: {type(response)}")
        
        # Check if the response is a StructuredQuestion
        if isinstance(response, StructuredQuestion):
            print("\nSuccessfully parsed into StructuredQuestion!")
            print(f"Intent: {response.intent}")
            print(f"Extracted classes: {response.extracted_classes}")
            print(f"Extracted entities: {response.extracted_entities}")
            print(f"Question steps: {response.question_steps}")
            return True
        else:
            print(f"\nError: Response is not a StructuredQuestion. Got {type(response)} instead.")
            print(f"Response content: {response}")
            return False
            
    except Exception as e:
        print(f"\nError occurred during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Run the async test function
    asyncio.run(test_structured_output_parsing())
