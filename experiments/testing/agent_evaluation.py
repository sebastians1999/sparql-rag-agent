import asyncio
import sys
import os
import json
import time
import datetime
from typing import Dict, List, Any
import pandas as pd

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import agent-related modules
from scr.agent.utils.graph import create_graph
from scr.agent.state.state import State
from langchain_core.messages import HumanMessage

# Import utilities for dataset handling
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../utilities')))
from format import process_federated_dataset, load_data_from_file


class AgentEvaluator:
    def __init__(self, dataset_path=None, output_dir=None, endpoint_sets=None):
        """
        Initialize the agent evaluator.
        
        Args:
            dataset_path: Path to the processed dataset JSON file
            output_dir: Directory to save evaluation results
            endpoint_sets: List of endpoint sets to test
        """
        self.endpoint_sets = endpoint_sets or ["Uniprot"]
        
        # Process the dataset if path not provided
        if dataset_path is None:
            print("Processing federated dataset...")
            self.output_dir = process_federated_dataset(
                endpoint_sets_to_test=self.endpoint_sets,
                output_dir=output_dir
            )
            # Find the most recent JSON file in the output directory
            json_files = [f for f in os.listdir(self.output_dir) if f.endswith('.json')]
            if not json_files:
                raise FileNotFoundError(f"No JSON files found in {self.output_dir}")
            
            self.dataset_path = os.path.join(self.output_dir, 'testset_meta_data.json')
        else:
            self.dataset_path = dataset_path
            self.output_dir = os.path.dirname(dataset_path)
        
        # Load the dataset
        print(f"Loading dataset from {self.dataset_path}")
        with open(self.dataset_path, 'r') as f:
            self.dataset = json.load(f)
        
        # Create the agent graph
        self.graph = create_graph()
        
        # Initialize results storage
        self.results = []
        
    async def run_single_test(self, question: str) -> Dict[str, Any]:
        """
        Run a single test with the given question and measure performance.
        
        Args:
            question: The natural language question to test
            
        Returns:
            Dictionary with test results and metrics
        """
        # Create initial state
        initial_state = State(
            messages=[HumanMessage(content=question)]
        )
        
        # Measure execution time
        start_time = time.time()
        
        # Run the agent
        try:
            final_state = await self.graph.ainvoke(initial_state)
            success = True
            error = None
        except Exception as e:
            final_state = None
            success = False
            error = str(e)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Extract token usage if available
        token_usage = {}
        if hasattr(final_state, 'token_usage'):
            token_usage = final_state.token_usage
        
        # Prepare result
        result = {
            "question": question,
            "success": success,
            "execution_time_seconds": execution_time,
            "token_usage": token_usage,
            "error": error,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Add the final answer if successful
        if success and final_state:
            # Extract the answer from the final state
            # Adjust this based on your State structure
            if hasattr(final_state, 'answer'):
                result["answer"] = final_state.answer
            elif hasattr(final_state, 'messages') and final_state.messages:
                # Get the last assistant message
                assistant_messages = [m for m in final_state.messages if hasattr(m, 'type') and m.type == 'assistant']
                if assistant_messages:
                    result["answer"] = assistant_messages[-1].content
        
        return result
    
    async def evaluate_dataset(self):
        """
        Evaluate the agent on all questions in the dataset.
        """
        print(f"Starting evaluation on {len(self.dataset)} questions...")
        
        for i, item in enumerate(self.dataset):
            question = item.get("natural_language_question")
            if not question:
                print(f"Skipping item {i} - no natural language question found")
                continue
                
            print(f"Testing question {i+1}/{len(self.dataset)}: {question[:50]}...")
            
            # Run the test
            result = await self.run_single_test(question)
            
            # Add metadata from the dataset item
            result["resource"] = item.get("resource")
            result["query"] = item.get("query")
            result["target_endpoint"] = item.get("target_endpoint")
            result["federates_with"] = item.get("federates_with")
            
            # Add to results
            self.results.append(result)
            
            # Print progress
            if result["success"]:
                status = f"✓ Success ({result['execution_time_seconds']:.2f}s)"
            else:
                status = f"✗ Failed: {result['error']}"
            print(f"  {status}")
            
        print(f"Evaluation complete. Tested {len(self.results)} questions.")
    
    def save_results(self):
        """
        Save the evaluation results to files.
        """
        # Create a timestamped directory for results
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        results_dir = os.path.join(self.output_dir, f"agent_eval_{timestamp}")
        os.makedirs(results_dir, exist_ok=True)
        
        # Save detailed JSON results
        json_path = os.path.join(results_dir, "evaluation_results.json")
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Create a summary DataFrame
        df = pd.DataFrame(self.results)
        
        # Calculate summary statistics
        summary = {
            "total_questions": len(self.results),
            "successful_runs": sum(1 for r in self.results if r["success"]),
            "failed_runs": sum(1 for r in self.results if not r["success"]),
            "average_execution_time": sum(r["execution_time_seconds"] for r in self.results) / len(self.results) if self.results else 0,
            "timestamp": timestamp
        }
        
        # Save summary
        summary_path = os.path.join(results_dir, "evaluation_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save as CSV for easy analysis
        csv_path = os.path.join(results_dir, "evaluation_results.csv")
        df.to_csv(csv_path, index=False)
        
        print(f"Results saved to {results_dir}")
        print(f"Success rate: {summary['successful_runs']}/{summary['total_questions']} ({summary['successful_runs']/summary['total_questions']*100:.1f}%)")
        print(f"Average execution time: {summary['average_execution_time']:.2f} seconds")
        
        return results_dir


async def main():
    """
    Main function to run the evaluation.
    """
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate the SPARQL RAG agent on a dataset")
    parser.add_argument("--dataset", help="Path to the processed dataset JSON file")
    parser.add_argument("--output", help="Directory to save evaluation results")
    parser.add_argument("--endpoints", nargs="+", default=["Uniprot"], 
                        help="List of endpoint sets to test (default: Uniprot)")
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = AgentEvaluator(
        dataset_path=args.dataset,
        output_dir=args.output,
        endpoint_sets=args.endpoints
    )
    
    # Run evaluation
    await evaluator.evaluate_dataset()
    
    # Save results
    evaluator.save_results()


if __name__ == "__main__":
    asyncio.run(main())
