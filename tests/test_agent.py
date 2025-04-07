import asyncio
import sys
import os
import time
from datetime import datetime
import dotenv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now your imports will work
from scr.agent.utils.graph import create_graph
from scr.agent.state.state import State
from langchain_core.messages import HumanMessage
from test_question import TEST_QUESTIONS
from langsmith import Client
from langchain_community.callbacks import get_openai_callback

# Load environment variables
dotenv.load_dotenv()

# Get LangSmith project name from environment
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "sparql-rag-agent")


async def run_test_with_langsmith(question: str) -> dict:
    """Run a test with LangSmith tracking."""
    # Create initial state
    initial_state = State(
        messages=[HumanMessage(content=question)]
    )

    # Create and run the graph with the provided config
    graph = create_graph()
    
    # Track start time
    start_time = time.time()
    
    # Run the agent
    final_state = await graph.ainvoke(initial_state)
    
    # Track end time
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Get LangSmith data
    client = Client()
    # List runs sorted by start time in descending order
    try:
        # Convert generator to list first
        runs_list = list(client.list_runs(
            project_name=LANGSMITH_PROJECT, 
            order_by="start_time", 
            order="desc"
        ))
        
        # Get the latest run
        latest_run = runs_list[0] if runs_list else None
        
        result = {
            "final_state": final_state,
            "execution_time": execution_time,
            "langsmith_data": None
        }
        
        if latest_run:
            # Safely calculate execution time, handling None values
            langsmith_execution_time = None
            if latest_run.end_time and latest_run.start_time:
                langsmith_execution_time = latest_run.end_time - latest_run.start_time
            
            # Safely get total tokens, defaulting to 0 if None
            total_tokens = latest_run.total_tokens if latest_run.total_tokens is not None else 0
            
            result["langsmith_data"] = {
                "execution_time": langsmith_execution_time,
                "total_tokens": total_tokens,
                "run_id": latest_run.id
            }
            
            # Print information, handling None values
            if langsmith_execution_time:
                print(f"LangSmith - Execution Time: {langsmith_execution_time}")
            else:
                print(f"LangSmith - Execution Time: Not available")
                
            print(f"LangSmith - Total Tokens: {total_tokens}")
        else:
            print("No runs found in LangSmith for the project.")
    except Exception as e:
        print(f"Error retrieving LangSmith data: {str(e)}")
        result = {
            "final_state": final_state,
            "execution_time": execution_time,
            "langsmith_data": None
        }
    
    return result


async def run_test_with_callback(question: str) -> dict:
    """Run a test with OpenAI callback tracking."""
    # Create initial state
    initial_state = State(
        messages=[HumanMessage(content=question)]
    )

    # Create and run the graph with the provided config
    graph = create_graph()
    
    # Track start time
    start_time = time.time()
    
    # Run with callback tracking
    with get_openai_callback() as cb:
        final_state = await graph.ainvoke(initial_state)
        
        callback_data = {
            "total_tokens": cb.total_tokens,
            "prompt_tokens": cb.prompt_tokens,
            "completion_tokens": cb.completion_tokens,
            "total_cost": cb.total_cost
        }
        
        print(f"OpenAI Callback - Total Tokens: {cb.total_tokens}")
        print(f"OpenAI Callback - Prompt Tokens: {cb.prompt_tokens}")
        print(f"OpenAI Callback - Completion Tokens: {cb.completion_tokens}")
        print(f"OpenAI Callback - Total Cost (USD): ${cb.total_cost}")
    
    # Track end time
    end_time = time.time()
    execution_time = end_time - start_time
    
    return {
        "final_state": final_state,
        "execution_time": execution_time,
        "callback_data": callback_data
    }


async def test_agent():
    """Run all test scenarios."""
    
    try:
        # Run tests for each question
        for scenario, question in TEST_QUESTIONS.items():
            print(f"\n\n=== Testing Scenario: {scenario} ===")
            print(f"Question: {question.strip()}")
            
            try:
                print("\n--- Method 1: LangSmith Tracking ---")
                langsmith_result = await run_test_with_langsmith(question)
                
                print("\n--- Method 2: OpenAI Callback Tracking ---")
                callback_result = await run_test_with_callback(question)
                
                # Compare results
                print("\n--- Comparison ---")
                print(f"LangSmith Execution Time: {langsmith_result['execution_time']:.2f} seconds")
                print(f"Callback Execution Time: {callback_result['execution_time']:.2f} seconds")
                
                if langsmith_result['langsmith_data']:
                    print(f"LangSmith Total Tokens: {langsmith_result['langsmith_data']['total_tokens']}")
                
                print(f"Callback Total Tokens: {callback_result['callback_data']['total_tokens']}")
                
                # Save results to file
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                results_dir = os.path.join(os.path.dirname(__file__), "../experiments/results")
                os.makedirs(results_dir, exist_ok=True)
                
                with open(os.path.join(results_dir, f"test_results_{scenario}_{timestamp}.txt"), "w") as f:
                    f.write(f"Scenario: {scenario}\n")
                    f.write(f"Question: {question.strip()}\n\n")
                    f.write(f"LangSmith Execution Time: {langsmith_result['execution_time']:.2f} seconds\n")
                    f.write(f"Callback Execution Time: {callback_result['execution_time']:.2f} seconds\n\n")
                    
                    if langsmith_result['langsmith_data']:
                        f.write(f"LangSmith Total Tokens: {langsmith_result['langsmith_data']['total_tokens']}\n")
                    
                    f.write(f"Callback Total Tokens: {callback_result['callback_data']['total_tokens']}\n")
                    f.write(f"Callback Prompt Tokens: {callback_result['callback_data']['prompt_tokens']}\n")
                    f.write(f"Callback Completion Tokens: {callback_result['callback_data']['completion_tokens']}\n")
                    f.write(f"Callback Total Cost (USD): ${callback_result['callback_data']['total_cost']}\n")
        
            except Exception as e:
                print(f"\nError in scenario '{scenario}': {str(e)}")
                continue
            
    except Exception as e:
        print(f"\nError during test execution: {str(e)}")
        raise


if __name__ == "__main__":
    # Run the tests
    asyncio.run(test_agent())