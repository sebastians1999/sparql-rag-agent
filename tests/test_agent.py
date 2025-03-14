import asyncio
from scr.agent.utils.graph import create_graph
from scr.agent.utils.config import Configuration, LLMConfig, RAGConfig
from scr.agent.utils.state.state import State
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

def print_test_results(final_state: State):
    """Print the results of a test run."""
    print("\n=== Test Results ===")
    print("\n1. Question Understanding:")
    print(f"Intent: {final_state.structured_question.intent}")
    print("\nExtracted Classes:")
    for cls in final_state.structured_question.extracted_classes:
        print(f"- {cls}")
    print("\nExtracted Entities:")
    for entity in final_state.structured_question.extracted_entities:
        print(f"- {entity}")
    
    print("\n2. Entity Resolution:")
    print(f"Retrieved Documents: {len(final_state.retrieved_docs)}")
    for doc in final_state.retrieved_docs:
        print(f"\nTitle: {doc.metadata['title']}")
        print(f"Type: {doc.metadata['type']}")
        print(f"Function: {doc.metadata['function']}")
        print(f"Content: {doc.page_content[:200]}...")
    
    print("\n3. SPARQL Query Construction:")
    print(f"Generated Query: {final_state.structured_output}")
    
    print("\n4. Steps Taken:")
    for step in final_state.steps:
        print(f"\nStep: {step.label}")
        print(f"Details: {step.details}")

async def test_agent():
    """Run all test scenarios."""
    
    try:
        # Run tests for each question
        for scenario, question in TEST_QUESTIONS.items():
            print(f"\n\n=== Testing Scenario: {scenario} ===")
            print(f"Question: {question.strip()}")
            
            try:
                final_state = await run_single_test(question)
                print_test_results(final_state)
            except Exception as e:
                print(f"\nError in scenario '{scenario}': {str(e)}")
                continue
            
    except Exception as e:
        print(f"\nError during test execution: {str(e)}")
        raise

if __name__ == "__main__":
    # Run the tests
    asyncio.run(test_agent())