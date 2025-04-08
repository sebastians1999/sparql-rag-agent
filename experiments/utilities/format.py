import re
import logging
from typing import Dict
import rdflib
import json
import os


logger = logging.getLogger(__name__)

RE_ART = re.compile(r'\b(a|an|the)\b')
RE_PUNC = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')


def remove_articles(_text):
    return RE_ART.sub(' ', _text)


def white_space_fix(_text):
    return ' '.join(_text.split())


def remove_punc(_text):
    return RE_PUNC.sub(' ', _text)  # convert punctuation to spaces


def lower(_text):
    return _text.lower()


def normalize(text):
    """Lower text and remove punctuation, articles and extra whitespace. """
    return white_space_fix(remove_articles(remove_punc(lower(text))))


sparql_query = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX sh: <http://www.w3.org/ns/shacl#>
PREFIX schema: <https://schema.org/>
PREFIX spex: <https://purl.expasy.org/sparql-examples/ontology#>

SELECT ?s ?comment ?select ?target ?federatesWith
WHERE {
    ?s rdfs:comment ?comment ;
       sh:select ?select ;
       schema:target ?target ;
       spex:federatesWith ?federatesWith .
}
"""


def get_sparql_question_meta(ttl_content: str) -> Dict[str, str]:
    g = rdflib.Graph()
    g.parse(data=ttl_content, format='turtle')
    results = g.query(sparql_query)

    print()

    for row in results:
        return {
            "resource": str(row.s),
            "natural_language_question": str(row.comment),
            "query": str(row.select),
            "target_endpoint": str(row.target),
            "federates_with": str(row.federatesWith),
        }


def load_data_from_file(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    # Convert list of dicts to dict of lists
    return {key: [d[key] for d in data] for key in data[0].keys()}


def process_federated_dataset(endpoint_sets_to_test=None, dataset_dir=None, output_dir=None) -> str:
    """
    Process all turtle files in the specified endpoint sets and store their metadata in a single JSON file.
    
    Args:
        endpoint_sets_to_test: List of endpoint sets to process (e.g., ["Uniprot", "Rhea"])
        dataset_dir: Directory containing the federated SPARQL dataset
        output_dir: Base directory for output files
        
    Returns:
        The directory path where the metadata file was stored
    """
    import os
    import json
    import datetime

    # Set defaults if not provided
    if endpoint_sets_to_test is None:
        endpoint_sets_to_test = ["Uniprot"]

    if dataset_dir is None:
        # Updated path to reflect new folder structure
        experiments_dir = os.path.dirname(os.path.dirname(__file__))
        dataset_dir = os.path.join(experiments_dir, "federated_sparql_dataset/examples_federated_02.04.2025")
    
    # Set the correct output directory (experiments/eval)
    if output_dir is None:
        # Get the experiments directory (parent of utilities)
        experiments_dir = os.path.dirname(os.path.dirname(__file__))
        output_dir = os.path.join(experiments_dir, "eval")
    
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Create timestamped directory for this evaluation run
    current_time = datetime.datetime.now()
    timestamped_dir_name = f"ev_{current_time.strftime('%Y-%m-%d_%H-%M-%S')}"
    timestamped_dir = os.path.join(output_dir, timestamped_dir_name)
    if not os.path.exists(timestamped_dir):
        os.makedirs(timestamped_dir)

    # Create a single list to store metadata for all endpoint sets
    all_metadata = []
    total_processed_count = 0

    # Process each endpoint set in the test list
    for endpoint_set in endpoint_sets_to_test:
        print(f"Processing endpoint set: {endpoint_set}")

        # Find the directory for this endpoint set (case-insensitive matching)
        endpoint_dir = None
        for item in os.listdir(dataset_dir):
            item_path = os.path.join(dataset_dir, item)
            if os.path.isdir(item_path) and item.lower() == endpoint_set.lower():
                endpoint_dir = item_path
                break

        if not endpoint_dir:
            print(f"Warning: Directory for endpoint set '{endpoint_set}' not found")
            continue

        # Process all .ttl files in this endpoint directory
        endpoint_processed_count = 0

        for root, dirs, files in os.walk(endpoint_dir):
            for file in files:
                if file.endswith(".ttl"):
                    file_path = os.path.join(root, file)

                    # Read the turtle file
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    # Extract metadata using the function from format.py
                    meta_data = get_sparql_question_meta(content)
                    if meta_data:
                        # Add source endpoint information to the metadata
                        meta_data["endpoint_set"] = endpoint_set
                        meta_data["file_path"] = os.path.relpath(file_path, dataset_dir)
                        meta_data["filename"] = os.path.basename(file_path)

                        all_metadata.append(meta_data)
                        endpoint_processed_count += 1
                        #print(f"Processed: {file_path}")

        total_processed_count += endpoint_processed_count
        #print(f"Processed {endpoint_processed_count} files from {endpoint_set}")

    # Save metadata only to the timestamped directory
    timestamped_output_file = os.path.join(timestamped_dir, 'testset_meta_data.json')
    with open(timestamped_output_file, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, indent=2, ensure_ascii=False)

    print(f"Saved metadata for {total_processed_count} files from {len(endpoint_sets_to_test)} endpoint sets to:")
    print(f"  - {timestamped_output_file}")

    return timestamped_dir


def process_specific_datasets_and_files(endpoint_files_map=None, dataset_dir=None, output_dir=None) -> str:
    """
    Process specific turtle files from specified endpoint sets and store their metadata in a single JSON file.
    
    Args:
        endpoint_files_map: Dictionary mapping endpoint sets to lists of specific TTL files to process
                           e.g., {"Uniprot": ["29.ttl", "20.ttl"], "Rhea": []}
                           Note: If an empty list is provided for a dataset, all files in that dataset will be processed
        dataset_dir: Directory containing the federated SPARQL dataset
        output_dir: Base directory for output files. If None, defaults to 'experiments/eval'
        
    Returns:
        The directory path where the metadata file was stored
    """
    import os
    import json
    import datetime

    # Set defaults if not provided
    if endpoint_files_map is None:
        endpoint_files_map = {"Uniprot": []}  # Default to Uniprot with all files (empty list)

    if dataset_dir is None:
        # Updated path to reflect new folder structure
        experiments_dir = os.path.dirname(os.path.dirname(__file__))
        dataset_dir = os.path.join(experiments_dir, "federated_sparql_dataset/examples_federated_02.04.2025")
    
    # Set the correct output directory
    if output_dir is None:
        # Default to experiments/eval if no output_dir is provided
        experiments_dir = os.path.dirname(os.path.dirname(__file__))
        output_dir = os.path.join(experiments_dir, "eval")
    elif not os.path.isabs(output_dir):
        # If a relative path is provided, make it relative to the experiments directory
        experiments_dir = os.path.dirname(os.path.dirname(__file__))
        output_dir = os.path.join(experiments_dir, output_dir)
    
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Create timestamped directory for this evaluation run
    current_time = datetime.datetime.now()
    timestamped_dir_name = f"ev_{current_time.strftime('%Y-%m-%d_%H-%M-%S')}"
    timestamped_dir = os.path.join(output_dir, timestamped_dir_name)
    if not os.path.exists(timestamped_dir):
        os.makedirs(timestamped_dir)

    # Create a single list to store metadata for all endpoint sets
    all_metadata = []
    total_processed_count = 0

    # Process each endpoint set in the map
    for endpoint_set, specific_files in endpoint_files_map.items():
        print(f"Processing endpoint set: {endpoint_set}")

        # Find the directory for this endpoint set (case-insensitive matching)
        endpoint_dir = None
        for item in os.listdir(dataset_dir):
            item_path = os.path.join(dataset_dir, item)
            if os.path.isdir(item_path) and item.lower() == endpoint_set.lower():
                endpoint_dir = item_path
                break

        if not endpoint_dir:
            print(f"Warning: Directory for endpoint set '{endpoint_set}' not found")
            continue

        # Process specific .ttl files in this endpoint directory
        endpoint_processed_count = 0

        for root, dirs, files in os.walk(endpoint_dir):
            for file in files:
                # Check if we should process this file:
                # - If specific_files is empty, process all .ttl files
                # - Otherwise, only process files that are in the specific_files list
                if file.endswith(".ttl") and (not specific_files or file in specific_files):
                    file_path = os.path.join(root, file)

                    # Read the turtle file
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    # Extract metadata using the function from format.py
                    meta_data = get_sparql_question_meta(content)
                    if meta_data:
                        # Add source endpoint information to the metadata
                        meta_data["endpoint_set"] = endpoint_set
                        meta_data["file_path"] = os.path.relpath(file_path, dataset_dir)
                        meta_data["filename"] = os.path.basename(file_path)

                        all_metadata.append(meta_data)
                        endpoint_processed_count += 1
                        print(f"Processed: {file}")

        total_processed_count += endpoint_processed_count
        print(f"Processed {endpoint_processed_count} files from {endpoint_set}")

    # Save metadata only to the timestamped directory
    timestamped_output_file = os.path.join(timestamped_dir, 'testset_meta_data.json')
    with open(timestamped_output_file, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, indent=2, ensure_ascii=False)

    print(f"Saved metadata for {total_processed_count} files from {len(endpoint_files_map)} endpoint sets to:")
    print(f"  - {timestamped_output_file}")

    return timestamped_dir


def save_queries_comparison(natural_language_question: str, ground_truth_query: str, predicted_query: str, file_path: str, file_name: str) -> None:
    """
    Creates a TTL file containing both the ground truth and predicted SPARQL queries for easy comparison.
    
    Args:
        natural_language_question: The natural language question
        predicted_query: The predicted SPARQL query
        file_path: Path to the directory where the TTL file should be saved
        file_name: Name of the file to create
    """
    # Create directory if it doesn't exist
    os.makedirs(file_path, exist_ok=True)
    
    # Combine directory path and filename
    full_path = os.path.join(file_path, file_name)
    
    # Format the content with clear separation and labels
    content = f"""
# Natural Language Question
# =======================
{natural_language_question}

# Ground Truth Query
# =================
{ground_truth_query}

# Predicted Query
# ==============
{predicted_query}
"""
    
    # Write to file
    with open(full_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Saved query comparison to: {full_path}")