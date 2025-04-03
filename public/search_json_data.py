#!/usr/bin/env python3
"""
Search Tool for Virginia Data Portal JSON Data

This script:
1. Connects to the Pinecone index containing embedded JSON data
2. Allows semantic search using OpenAI's API
3. Returns relevant JSON data based on the search query
"""

import os
import json
import argparse
import logging
from pathlib import Path
import openai
import pinecone
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("search_json.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("json_searcher")

# Default OpenAI model for embeddings
EMBEDDING_MODEL = "text-embedding-3-small"

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Search embedded JSON data using semantic search')
    parser.add_argument('--query', required=True, help='Search query')
    parser.add_argument('--api-key', required=False, help='OpenAI API key')
    parser.add_argument('--pinecone-key', required=False, help='Pinecone API key')
    parser.add_argument('--pinecone-env', default='us-east-1-aws', help='Pinecone environment')
    parser.add_argument('--index-name', default='virginia-data-portal', help='Pinecone index name')
    parser.add_argument('--top-k', type=int, default=5, help='Number of results to return')
    parser.add_argument('--output', help='Output file for results (JSON format)')
    return parser.parse_args()

def get_embedding(text, model=EMBEDDING_MODEL):
    """Get embedding from OpenAI API."""
    try:
        response = openai.Embedding.create(
            model=model,
            input=text
        )
        return response["data"][0]["embedding"]
    except Exception as e:
        logger.error(f"Error getting embedding: {str(e)}")
        return None

def connect_to_pinecone(api_key, environment, index_name):
    """Connect to Pinecone index."""
    try:
        # Initialize Pinecone with the new approach
        pc = pinecone.Pinecone(api_key=api_key)
        
        # Connect to index
        index = pc.Index(index_name)
        return index
    except Exception as e:
        logger.error(f"Error connecting to Pinecone: {str(e)}")
        return None

def search_json_data(index, query, top_k=5):
    """Search JSON data using semantic search."""
    # Get embedding for the query
    query_embedding = get_embedding(query)
    if not query_embedding:
        logger.error("Failed to get embedding for query")
        return []
    
    # Search Pinecone
    try:
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        # Process results
        processed_results = []
        for match in results.matches:
            processed_results.append({
                'score': match.score,
                'file_name': match.metadata.get('file_name', 'Unknown'),
                'chunk_index': match.metadata.get('chunk_index', 0),
                'text': match.metadata.get('text', '')
            })
        
        return processed_results
    
    except Exception as e:
        logger.error(f"Error searching Pinecone: {str(e)}")
        return []

def format_results_for_chatgpt(results):
    """Format search results for ChatGPT."""
    formatted_results = []
    
    for result in results:
        formatted_results.append({
            'file': result['file_name'],
            'relevance_score': result['score'],
            'content': result['text']
        })
    
    return formatted_results

def main():
    """Main function to search JSON data."""
    args = parse_arguments()
    
    # Get API keys from arguments or environment
    openai_api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    pinecone_api_key = args.pinecone_key or os.environ.get("PINECONE_API_KEY")
    
    if not openai_api_key:
        logger.error("OpenAI API key not provided. Please set OPENAI_API_KEY environment variable or use --api-key.")
        return
    
    if not pinecone_api_key:
        logger.error("Pinecone API key not provided. Please set PINECONE_API_KEY environment variable or use --pinecone-key.")
        return
    
    # Initialize OpenAI
    openai.api_key = openai_api_key
    
    # Connect to Pinecone
    index = connect_to_pinecone(
        api_key=pinecone_api_key,
        environment=args.pinecone_env,
        index_name=args.index_name
    )
    
    if not index:
        logger.error("Failed to connect to Pinecone")
        return
    
    # Search JSON data
    results = search_json_data(index, args.query, top_k=args.top_k)
    
    if not results:
        print("No results found.")
        return
    
    # Format results for ChatGPT
    formatted_results = format_results_for_chatgpt(results)
    
    # Print results
    print(f"\n=== Search Results for: '{args.query}' ===")
    for i, result in enumerate(formatted_results, 1):
        print(f"\n--- Result {i} (Score: {result['relevance_score']:.4f}) ---")
        print(f"File: {result['file']}")
        print(f"Content: {result['content'][:500]}...")  # Show first 500 chars
    
    # Save results to file if requested
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(formatted_results, f, indent=2)
        print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main() 