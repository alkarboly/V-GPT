"""Script to simulate n8n workflow for testing purposes.

This script simulates the n8n workflow by:
1. Starting the API server in a separate process
2. Sending requests to the API with test queries
3. Processing the responses in the same way n8n would
"""

import os
import sys
import json
import time
import requests
import subprocess
from pathlib import Path
from typing import Dict, Any, List

# Add the project root to the Python path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.append(str(project_root))

def start_api_server():
    """Start the API server as a subprocess."""
    print("Starting API server...")
    
    # Use Python executable to run the API server
    python_executable = sys.executable
    
    # Start the server as a separate process
    api_process = subprocess.Popen(
        [python_executable, "-m", "src.main", "api"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for server to start
    print("Waiting for API server to start...")
    time.sleep(5)
    
    return api_process

def test_api_connection():
    """Test if API server is running and accessible."""
    try:
        response = requests.get("http://localhost:8000/")
        if response.status_code == 200:
            print("API server is running!")
            return True
        else:
            print(f"API server returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("Could not connect to API server!")
        return False

def simulate_n8n_search(query: str, limit: int = 5) -> Dict[str, Any]:
    """Simulate n8n workflow search request.
    
    Args:
        query: Search query
        limit: Number of results to return
        
    Returns:
        Processed search results
    """
    print(f"\n=== Simulating n8n search for: '{query}' ===")
    
    # Prepare the request payload (same as in n8n workflow)
    payload = {
        "query": query,
        "limit": limit
    }
    
    # Send request to API server
    response = requests.post(
        "http://localhost:8000/search",
        json=payload
    )
    
    if response.status_code != 200:
        print(f"Error: API returned status code {response.status_code}")
        print(response.text)
        return {"error": response.text}
    
    # Parse response
    results = response.json()["results"]
    
    # Format results (simulating the n8n Format Results node)
    formatted_results = []
    
    for i in range(len(results["ids"])):
        result_id = results["ids"][i]
        metadata = results["metadatas"][i]
        
        formatted_result = {
            "id": result_id,
            "title": metadata.get("title", "No title"),
            "description": metadata.get("description", "No description"),
            "tags": metadata.get("tags", []),
            "organization": metadata.get("organization", "Unknown"),
            "url": metadata.get("source_page", ""),
            "similarity_score": results["distances"][i] if "distances" in results else 0
        }
        
        formatted_results.append(formatted_result)
    
    # Create final output (simulating n8n output)
    output = {
        "query": query,
        "results_count": len(formatted_results),
        "results": formatted_results
    }
    
    # Print results summary
    print(f"Found {output['results_count']} results:")
    for i, result in enumerate(output["results"]):
        print(f"  {i+1}. {result['title']}")
        if result.get("description"):
            desc = result["description"]
            print(f"     {desc[:100]}..." if len(desc) > 100 else f"     {desc}")
        print(f"     Similarity: {result['similarity_score']}")
    
    return output

def run_simulation():
    """Run the complete n8n workflow simulation."""
    api_process = None
    
    try:
        # Start API server
        api_process = start_api_server()
        
        # Test connection
        if not test_api_connection():
            print("Failed to connect to API server. Aborting simulation.")
            return False
        
        # Run test searches
        search_queries = [
            "education funding",
            "healthcare data",
            "transportation infrastructure",
            "environmental protection",
            "public safety"
        ]
        
        for query in search_queries:
            result = simulate_n8n_search(query)
            
            # Verify we got results
            if result.get("error"):
                print(f"Search failed for query: {query}")
                return False
            elif result["results_count"] == 0:
                print(f"Warning: No results found for query: {query}")
            
        print("\n=== n8n Workflow Simulation Completed Successfully ===")
        return True
        
    except Exception as e:
        print(f"Error during simulation: {str(e)}")
        return False
        
    finally:
        # Always terminate the API server
        if api_process:
            print("\nStopping API server...")
            api_process.terminate()
            try:
                api_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                api_process.kill()

if __name__ == "__main__":
    success = run_simulation()
    sys.exit(0 if success else 1) 