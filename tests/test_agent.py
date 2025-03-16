import asyncio
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now your imports will work
from scr.agent.utils.graph import create_graph
from scr.agent.state.state import State
from langchain_core.messages import HumanMessage
from test_question import TEST_QUESTIONS


async def run_single_test(question: str) -> State:
    """Run a single test with the given question and configuration."""
    # Create initial state
    initial_state = State(
        messages=[HumanMessage(content=question)]
    )

    # Create and run the graph with the provided config
    graph = create_graph()
    return await graph.ainvoke(initial_state)

async def test_agent():
    """Run all test scenarios."""
    
    try:
        # Run tests for each question
        for scenario, question in TEST_QUESTIONS.items():
            print(f"\n\n=== Testing Scenario: {scenario} ===")
            print(f"Question: {question.strip()}")
            
            try:
                final_state = await run_single_test(question)
        
            except Exception as e:
                print(f"\nError in scenario '{scenario}': {str(e)}")
                continue
            
    except Exception as e:
        print(f"\nError during test execution: {str(e)}")
        raise

if __name__ == "__main__":
    # Run the tests
    asyncio.run(test_agent())