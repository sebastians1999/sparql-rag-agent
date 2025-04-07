import asyncio
import sys
import os
import time
from datetime import datetime
import dotenv
from experiments.utilities.format import process_federated_dataset, load_data_from_file, process_specific_datasets_and_files, save_queries_comparison
from datasets import Dataset
from scr.agent.state.state import State
from langchain_core.messages import HumanMessage
from scr.agent.utils.graph import create_graph
from langsmith import Client
from datasets import Dataset
from experiments.utilities.metrics import eval_pairs
import nltk
import json




class AgentEvaluator:
    def __init__(self, dataset_path=None, output_dir=None, endpoint_sets=None, project_name_langsmith: str ="sparql-rag-agent", test: bool = False):
        
        self.endpoint_sets = endpoint_sets
        self.output_dir = process_specific_datasets_and_files(self.endpoint_sets)
        self.test_dataset_path = os.path.join(self.output_dir, 'testset_meta_data.json')
        self.test_dataset = Dataset.from_dict(load_data_from_file(self.test_dataset_path))
        self.graph = create_graph()
        self.client = Client()
        self.project_name_langsmith = project_name_langsmith
        self.results = []
        self.evaluation_dataset_path = os.path.join(self.output_dir, 'evaluation_dataset.json')
        self.test = test

        
    async def run_single_test(self, question: str) -> State:
        # Create initial state

        try:    
            initial_state = State(
                messages=[HumanMessage(content=question)]
            )
            
            final_state = await self.graph.ainvoke(initial_state)
            print(final_state)
        except Exception as e:
            print(f"Error processing question in agent: {str(e)}")
            return None

        runs = self.client.list_runs(project_name=self.project_name_langsmith, is_root=True)
        first_run = next(runs)
        print("Got first run!")

        # Handle None values for datetime fields
        execution_time = ""
        if first_run.end_time is not None and first_run.start_time is not None:
            execution_time = first_run.end_time - first_run.start_time
        else:
            print("Warning: Run has None value for start_time or end_time")
            execution_time = 0  # Default value

        # Get child runs to find sparql_query_construction run
        child_runs = list(self.client.list_runs(project_name=self.project_name_langsmith, parent_run_id=first_run.id))
        
        # Filter to find the sparql_query_construction run
        sparql_construction_run = None
        for run in child_runs:
            if run.name == "sparql_query_construction":
                sparql_construction_run = run
                break

        if sparql_construction_run:
            result = {
                "final_state_response": final_state.get("structured_output", {}).get("query", ""),
                "run_id_langsmith": str(first_run.id),
                "in_dataset": first_run.in_dataset,
                "execution_time": str(execution_time),
                # Add the specific metrics for the sparql_query_construction run
                "sparql_construction_prompt_tokens": sparql_construction_run.prompt_tokens or 0,
                "sparql_construction_completion_tokens": sparql_construction_run.completion_tokens or 0,
                "sparql_construction_total_tokens": sparql_construction_run.total_tokens or 0,
                "sparql_construction_prompt_cost": sparql_construction_run.prompt_cost or 0,
                "sparql_construction_completion_cost": sparql_construction_run.completion_cost or 0,
                "sparql_construction_total_cost": sparql_construction_run.total_cost or 0,
                # Keep the original metrics too
                "prompt_tokens": first_run.prompt_tokens or 0,
                "completion_tokens": first_run.completion_tokens or 0,
                "total_tokens": first_run.total_tokens or 0,
                "prompt_cost": first_run.prompt_cost or 0,
                "completion_cost": first_run.completion_cost or 0,
                "total_cost": first_run.total_cost or 0,
            }
        else:
            print("Warning: Could not find the sparql_query_construction run")
            # Fall back to using the parent run metrics
            result = {
                "final_state_response": final_state.get("structured_output", {}).get("query", ""),
                "run_id_langsmith": str(first_run.id),
                "in_dataset": first_run.in_dataset,
                "execution_time": str(execution_time),
                "prompt_tokens": first_run.prompt_tokens or 0,
                "completion_tokens": first_run.completion_tokens or 0,
                "total_tokens": first_run.total_tokens or 0,
                "prompt_cost": first_run.prompt_cost or 0,
                "completion_cost": first_run.completion_cost or 0,
                "total_cost": first_run.total_cost or 0,
            }
            
        print("Result:", result)

        return result

    async def run_all_tests(self):

        updated_results = []
        
        if self.test:
            test_dataset = self.test_dataset.select(range(1))
        else:
            test_dataset = self.test_dataset

        for i, item in enumerate(test_dataset):
            # Access the fields directly since we're using a Dataset object
            question = item["natural_language_question"]
            #print(question)
            if not question:
                print(f"Skipping item {i+1}/{len(test_dataset)} - no natural language question found")
                continue
            
            try:
                print(f"Sending question {i+1}/{len(test_dataset)} to agent...")
                result = await self.run_single_test(question)
                print("Got result")
                
                updated_item = {
                    # Meta data - access fields directly
                    "resource": item.get("resource", ""),
                    "natural_language_question": question,
                    "ground_truth_query": item.get("query", ""),
                    "target_endpoint": item.get("target_endpoint", ""),
                    "federates_with": item.get("federates_with", ""),
                    "endpoint_set": item.get("endpoint_set", ""),
                    "file_path": item.get("file_path", ""),
                    
                    # New data
                    "predicted_query": result["final_state_response"],
                    "run_id_langsmith": str(result["run_id_langsmith"]),
                    "in_dataset": result["in_dataset"],
                    "execution_time": str(result["execution_time"]),
                    "prompt_tokens": result["prompt_tokens"],
                    "completion_tokens": result["completion_tokens"],
                    "total_tokens": result["total_tokens"],
                    "prompt_cost": result["prompt_cost"],
                    "completion_cost": result["completion_cost"],
                    "total_cost": result["total_cost"],
                    "evaluation_timestamp": datetime.now().isoformat()
                }
                
                if "sparql_construction_prompt_tokens" in result:
                    updated_item["sparql_construction_prompt_tokens"] = result["sparql_construction_prompt_tokens"]
                    updated_item["sparql_construction_completion_tokens"] = result["sparql_construction_completion_tokens"]
                    updated_item["sparql_construction_total_tokens"] = result["sparql_construction_total_tokens"]
                    updated_item["sparql_construction_prompt_cost"] = result["sparql_construction_prompt_cost"]
                    updated_item["sparql_construction_completion_cost"] = result["sparql_construction_completion_cost"]
                    updated_item["sparql_construction_total_cost"] = result["sparql_construction_total_cost"]
                
                updated_results.append(updated_item)
            except Exception as e:
                print(f"Error processing question {i+1}/{len(test_dataset)}: {str(e)}")
        
        self.updated_dataset = Dataset.from_list(updated_results)

        # # Download punkt tokenizer if not already downloaded
        # try:
        #     nltk.data.find('tokenizers/punkt')
        # except LookupError:
        #     nltk.download('punkt')  

        # for item in self.updated_dataset:
        #     item["metrics"] = eval_pairs(zip(item["ground_truth_query"], item["predicted_query"]))

        # TODO: Implement sparql validation tool

        with open(self.evaluation_dataset_path, 'w', encoding='utf-8') as f:
            json.dump(self.updated_dataset.to_list(), f, indent=2, ensure_ascii=False)

        for item in self.updated_dataset:
            # Get the filename from the item or create one based on the resource
            if "filename" in item:
                filename = os.path.splitext(item["filename"])[0] + "_comparison.ttl"
            else:
                resource_id = item.get("resource", "").split("/")[-1]
                filename = f"{resource_id}_comparison.ttl"
            
            # Save the comparison file
            save_queries_comparison(
                item.get("natural_language_question", ""),
                item.get("ground_truth_query", ""), 
                item.get("predicted_query", ""), 
                self.output_dir,
                filename
            )