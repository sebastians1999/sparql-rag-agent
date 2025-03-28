from entity_indexing_pipeline_v3 import EmbeddingPipeline
import ray

def test_pipeline():
    """Test the pipeline with example documents"""
    print("Initializing EmbeddingPipeline...")
    pipeline = EmbeddingPipeline()
        
    sample_docs = [
            {
                "uri": "http://example.org/disease/1",
                "label": "Alzheimer's Disease",
                "type": "disease",
                "description": "A progressive neurological disorder that causes brain cells to die"
            },
            {
                "uri": "http://example.org/disease/2",
                "label": "Parkinson's Disease",
                "type": "disease",
                "description": "A brain disorder that causes unintended or uncontrollable movements"
            },
            {
                "uri": "http://example.org/disease/3",
                "label": "Multiple Sclerosis",
                "type": "disease",
                "description": "A disease that affects the central nervous system"
            }
        ]
        
    print("\nAdding sample documents...")
    pipeline.add_documents(sample_docs)
        
    # test_queries = [
    #         "neurodegenerative disease",
    #         "brain disorder",
    #         "nervous system condition"
    #     ]
        
    # for query in test_queries:
    #     print(f"\nTesting search for query: '{query}'")
    #     results = pipeline.search(query)
            
    #     print("\nSearch Results:")
    #     for idx, result in enumerate(results, 1):
    #         print(f"\n{idx}. Score: {result.score:.4f}")
    #         print(f"   Label: {result.payload['label']}")
    #         print(f"   Type: {result.payload['type']}")
    #         if 'description' in result.payload:
    #             print(f"   Description: {result.payload['description']}")

if __name__ == "__main__":


    ray.init(ignore_reinit_error=True)

    test_pipeline()

    ray.shutdown()