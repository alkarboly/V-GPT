"""Test script for verifying n8n integration with vector database.

This script:
1. Creates 5 sample datasets
2. Uploads them to the vector database 
3. Verifies they were stored correctly
4. Performs a test search
5. Deletes the test embeddings
"""

import os
import sys
import json
import uuid
from pathlib import Path
from typing import List, Dict, Any

# Add the project root to the Python path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.append(str(project_root))

from src.metadata_extractor import VirginiaDataPortalExtractor
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def create_test_datasets(count: int = 5) -> List[Dict[str, Any]]:
    """Create sample datasets for testing.
    
    Args:
        count: Number of datasets to create
        
    Returns:
        List of sample datasets
    """
    datasets = []
    
    topics = [
        "Education", "Healthcare", "Transportation", 
        "Environment", "Finance", "Public Safety"
    ]
    
    for i in range(count):
        # Create unique ID using uuid
        dataset_id = f"test-dataset-{uuid.uuid4()}"
        
        # Create dataset with varied topics
        dataset = {
            'id': dataset_id,
            'title': f"Test Dataset {i+1} - {topics[i % len(topics)]} Data",
            'description': f"This is a test dataset about {topics[i % len(topics)].lower()} "
                           f"created for testing the n8n integration with vector database.",
            'tags': [topics[i % len(topics)], "test", "n8n-integration"],
            'urls': [f"https://example.com/test-dataset-{i+1}"],
            'resource_count': 1,
            'organization': "Test Organization",
            'source_page': f"https://data.virginia.gov/dataset/test-dataset-{i+1}"
        }
        
        datasets.append(dataset)
    
    return datasets

def upload_test_datasets(extractor: VirginiaDataPortalExtractor, 
                        datasets: List[Dict[str, Any]]) -> None:
    """Upload test datasets to the vector database.
    
    Args:
        extractor: The initialized extractor instance
        datasets: List of datasets to upload
    """
    print("\n=== Uploading Test Datasets ===")
    
    for dataset in datasets:
        # Generate text for embedding (same logic as in the embedding pipeline)
        text_for_embedding = f"{dataset['title']} {dataset['description']} {' '.join(dataset['tags'])}"
        
        # Generate embedding
        embedding = extractor.generate_embedding(text_for_embedding)
        
        # Store in ChromaDB
        extractor.collection.add(
            embeddings=[embedding],
            documents=[text_for_embedding],
            metadatas=[dataset],
            ids=[dataset['id']]
        )
        
        print(f"Uploaded: {dataset['id']} - {dataset['title']}")

def verify_datasets_exist(extractor: VirginiaDataPortalExtractor, 
                         dataset_ids: List[str]) -> bool:
    """Verify that datasets exist in the vector database.
    
    Args:
        extractor: The initialized extractor instance
        dataset_ids: List of dataset IDs to check
        
    Returns:
        True if all datasets exist, False otherwise
    """
    print("\n=== Verifying Datasets Exist ===")
    
    # Get all dataset IDs from the collection
    all_ids = extractor.collection.get()["ids"]
    
    # Check if each test dataset ID exists
    missing_ids = []
    for dataset_id in dataset_ids:
        if dataset_id not in all_ids:
            missing_ids.append(dataset_id)
    
    if missing_ids:
        print(f"WARNING: {len(missing_ids)} datasets are missing from the database:")
        for missing_id in missing_ids:
            print(f"  - {missing_id}")
        return False
    else:
        print(f"SUCCESS: All {len(dataset_ids)} test datasets exist in the database.")
        return True

def test_search(extractor: VirginiaDataPortalExtractor, 
               test_query: str,
               expected_topic: str) -> bool:
    """Test search functionality.
    
    Args:
        extractor: The initialized extractor instance
        test_query: Query to search for
        expected_topic: Topic that should be found
        
    Returns:
        True if search works correctly, False otherwise
    """
    print(f"\n=== Testing Search with query: '{test_query}' ===")
    
    # Perform search
    results = extractor.search_similar_datasets(test_query, n_results=5)
    
    # Print results
    print(f"Found {len(results['ids'])} results:")
    found_expected_topic = False
    
    for i in range(len(results['ids'])):
        result_id = results['ids'][i]
        metadata = results['metadatas'][i]
        
        # Check if the expected topic is in the result
        if expected_topic.lower() in metadata['title'].lower() or expected_topic.lower() in metadata['description'].lower():
            found_expected_topic = True
        
        print(f"  {i+1}. {metadata['title']} (ID: {result_id})")
        print(f"     Tags: {', '.join(metadata['tags'])}")
        print(f"     Similarity: {results['distances'][i]}")
    
    if found_expected_topic:
        print(f"SUCCESS: Found results related to '{expected_topic}'")
        return True
    else:
        print(f"WARNING: No results found related to '{expected_topic}'")
        return False

def delete_test_datasets(extractor: VirginiaDataPortalExtractor, 
                       dataset_ids: List[str]) -> None:
    """Delete test datasets from the vector database.
    
    Args:
        extractor: The initialized extractor instance
        dataset_ids: List of dataset IDs to delete
    """
    print("\n=== Deleting Test Datasets ===")
    
    # Delete all test datasets
    extractor.collection.delete(ids=dataset_ids)
    
    print(f"Deleted {len(dataset_ids)} test datasets.")
    
    # Verify they were deleted
    all_ids = extractor.collection.get()["ids"]
    remaining_test_ids = [dataset_id for dataset_id in dataset_ids if dataset_id in all_ids]
    
    if remaining_test_ids:
        print(f"WARNING: {len(remaining_test_ids)} test datasets were not deleted:")
        for remaining_id in remaining_test_ids:
            print(f"  - {remaining_id}")
    else:
        print("SUCCESS: All test datasets were successfully deleted.")

def run_test():
    """Run the complete test suite."""
    print("=== Starting n8n Integration Test ===\n")
    
    # Initialize the extractor
    extractor = VirginiaDataPortalExtractor()
    
    # Create test datasets
    datasets = create_test_datasets(5)
    dataset_ids = [dataset['id'] for dataset in datasets]
    
    try:
        # Upload test datasets
        upload_test_datasets(extractor, datasets)
        
        # Verify datasets exist
        datasets_exist = verify_datasets_exist(extractor, dataset_ids)
        
        # Test search for education
        education_search = test_search(extractor, "education funding", "Education")
        
        # Test search for healthcare
        healthcare_search = test_search(extractor, "healthcare medical", "Healthcare")
        
        # Report test results
        print("\n=== Test Summary ===")
        print(f"Datasets Exist Test: {'PASSED' if datasets_exist else 'FAILED'}")
        print(f"Education Search Test: {'PASSED' if education_search else 'FAILED'}")
        print(f"Healthcare Search Test: {'PASSED' if healthcare_search else 'FAILED'}")
        
        test_success = datasets_exist and education_search and healthcare_search
        print(f"\nOverall Test: {'PASSED' if test_success else 'FAILED'}")
    
    finally:
        # Always delete test datasets, even if tests fail
        delete_test_datasets(extractor, dataset_ids)
    
    print("\n=== Test Completed ===")
    return test_success

if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1) 