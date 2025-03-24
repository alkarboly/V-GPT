#!/usr/bin/env python3
"""
Embeddings Generator for Virginia Data Portal

This script:
1. Loads the JSON data created by prepare_json.py
2. Generates embeddings using OpenAI's API
3. Stores the embeddings in Pinecone vector database
"""

import os
import json
import time
from pathlib import Path
import logging
import argparse
from tqdm import tqdm
import openai
import pinecone

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("embed_store.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("embedding_generator")

# Set up paths
script_path = Path(__file__).resolve().parent
project_root = script_path.parent
json_path = project_root / "data" / "json"

# Default OpenAI model for embeddings
EMBEDDING_MODEL = "text-embedding-ada-002"
EMBEDDING_DIMENSION = 1536  # Dimension for text-embedding-ada-002

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate embeddings and store in Pinecone')
    parser.add_argument('--api-key', required=False, help='OpenAI API key')
    parser.add_argument('--pinecone-key', required=False, help='Pinecone API key')
    parser.add_argument('--pinecone-env', default='us-west1-gcp', help='Pinecone environment')
    parser.add_argument('--index-name', default='virginia-data', help='Pinecone index name')
    parser.add_argument('--batch-size', type=int, default=50, help='Batch size for processing')
    return parser.parse_args()

def get_embedding(text, model=EMBEDDING_MODEL, max_retries=3, retry_delay=1):
    """Get embedding from OpenAI API with retry logic."""
    if not text or text.strip() == "":
        return None
    
    for attempt in range(max_retries):
        try:
            response = openai.Embedding.create(
                model=model,
                input=text
            )
            return response["data"][0]["embedding"]
        except Exception as e:
            logger.warning(f"Error getting embedding (attempt {attempt+1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (2**attempt))  # Exponential backoff
            else:
                logger.error(f"Failed to get embedding after {max_retries} attempts")
                return None

def generate_dataset_embeddings(dataset):
    """Generate embeddings for a dataset and its columns."""
    embeddings = []
    
    # Prepare dataset metadata for embedding
    dataset_id = dataset.get('id', '')
    dataset_title = dataset.get('title', '')
    dataset_desc = dataset.get('description', '')
    organization = dataset.get('organization', '')
    tags = ', '.join(dataset.get('tags', []))
    
    # Create text for dataset level embedding
    dataset_text = f"Dataset: {dataset_title}\nDescription: {dataset_desc}\nOrganization: {organization}\nTags: {tags}"
    
    # Get dataset embedding
    dataset_embedding = get_embedding(dataset_text)
    if dataset_embedding:
        embeddings.append({
            'id': f"{dataset_id}_metadata",
            'text': dataset_text,
            'embedding': dataset_embedding,
            'metadata': {
                'type': 'dataset',
                'dataset_id': dataset_id,
                'title': dataset_title,
                'organization': organization,
                'tags': dataset.get('tags', [])
            }
        })
    
    # Process each column
    for i, column in enumerate(dataset.get('columns', [])):
        column_name = column.get('name', '')
        column_type = column.get('type', '')
        column_desc = column.get('description', '')
        column_samples = column.get('sample_values', '')
        
        # Create text for column embedding
        column_text = f"Dataset: {dataset_title}\nColumn: {column_name}\nType: {column_type}\nDescription: {column_desc}\nSample Values: {column_samples}"
        
        # Get column embedding
        column_embedding = get_embedding(column_text)
        if column_embedding:
            embeddings.append({
                'id': f"{dataset_id}_column_{i}",
                'text': column_text,
                'embedding': column_embedding,
                'metadata': {
                    'type': 'column',
                    'dataset_id': dataset_id,
                    'dataset_title': dataset_title,
                    'column_name': column_name,
                    'column_type': column_type,
                    'organization': organization
                }
            })
    
    return embeddings

def initialize_pinecone(api_key, environment, index_name):
    """Initialize Pinecone client and create index if it doesn't exist."""
    pinecone.init(api_key=api_key, environment=environment)
    
    # Check if index exists
    existing_indexes = pinecone.list_indexes()
    
    if index_name not in existing_indexes:
        logger.info(f"Creating new Pinecone index: {index_name}")
        pinecone.create_index(
            name=index_name,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine"
        )
        # Wait for index to be ready
        time.sleep(1)
    
    # Connect to index
    index = pinecone.Index(index_name)
    
    return index

def store_embeddings_in_pinecone(index, embeddings, batch_size=100):
    """Store embeddings in Pinecone index using batching."""
    total_embeddings = len(embeddings)
    logger.info(f"Storing {total_embeddings} embeddings in Pinecone")
    
    # Process in batches
    for i in range(0, total_embeddings, batch_size):
        batch = embeddings[i:i+batch_size]
        
        # Convert to Pinecone format
        pinecone_batch = []
        for item in batch:
            pinecone_batch.append({
                'id': item['id'],
                'values': item['embedding'],
                'metadata': {
                    'text': item['text'],
                    **item['metadata']
                }
            })
        
        # Upsert batch
        index.upsert(vectors=pinecone_batch)
        logger.info(f"Stored batch {i//batch_size + 1}/{(total_embeddings+batch_size-1)//batch_size}")
        
        # Sleep to avoid rate limits
        time.sleep(0.5)

def main():
    """Main function to process JSON data, generate embeddings and store in Pinecone."""
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
    
    # Initialize Pinecone
    index = initialize_pinecone(
        api_key=pinecone_api_key,
        environment=args.pinecone_env,
        index_name=args.index_name
    )
    
    # Load the search index
    search_index_path = json_path / "search_index.json"
    
    if not search_index_path.exists():
        logger.error(f"Search index file not found at {search_index_path}")
        logger.error("Please run prepare_json.py first to generate the search index.")
        return
    
    with open(search_index_path, 'r', encoding='utf-8') as f:
        datasets = json.load(f)
    
    logger.info(f"Loaded {len(datasets)} datasets from search index")
    
    # Generate embeddings for all datasets
    all_embeddings = []
    
    for dataset in tqdm(datasets, desc="Generating embeddings"):
        dataset_embeddings = generate_dataset_embeddings(dataset)
        all_embeddings.extend(dataset_embeddings)
        
        # Small delay to avoid OpenAI rate limits
        time.sleep(0.1)
    
    logger.info(f"Generated {len(all_embeddings)} embeddings in total")
    
    # Store embeddings in Pinecone
    store_embeddings_in_pinecone(index, all_embeddings, batch_size=args.batch_size)
    
    # Print stats from Pinecone
    stats = index.describe_index_stats()
    logger.info(f"Pinecone index stats: {stats}")
    
    print("\n=== Embedding and Storage Complete ===")
    print(f"Total embeddings created and stored: {len(all_embeddings)}")
    print(f"Pinecone index: {args.index_name}")
    print(f"Vector count in index: {stats['total_vector_count']}")

if __name__ == "__main__":
    main() 