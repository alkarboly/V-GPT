#!/usr/bin/env python3
"""
JSON Files Embedder for Virginia Data Portal

This script:
1. Loads all JSON files from the data/json directory
2. Generates embeddings using OpenAI's API
3. Stores the embeddings in Pinecone vector database for semantic search
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
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("embed_json.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("json_embedder")

# Set up paths
script_path = Path(__file__).resolve().parent
project_root = script_path.parent
json_path = project_root / "data" / "json"

# Default OpenAI model for embeddings
EMBEDDING_MODEL = "text-embedding-3-small"  # Using the latest model
EMBEDDING_DIMENSION = 1536  # Dimension for text-embedding-3-small

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate embeddings for JSON files and store in Pinecone')
    parser.add_argument('--api-key', required=False, help='OpenAI API key')
    parser.add_argument('--pinecone-key', required=False, help='Pinecone API key')
    parser.add_argument('--pinecone-env', default='us-east-1-aws', help='Pinecone environment')
    parser.add_argument('--index-name', default='virginia-data-portal', help='Pinecone index name')
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size for processing')
    parser.add_argument('--chunk-size', type=int, default=1000, help='Maximum characters per chunk')
    parser.add_argument('--num-files', type=int, default=None, help='Number of files to process (default: all files)')
    return parser.parse_args()

def get_embedding(text, model=EMBEDDING_MODEL, max_retries=3, retry_delay=1):
    """Get embedding from OpenAI API with retry logic."""
    if not text or text.strip() == "":
        logger.warning("Empty text provided for embedding")
        return None
    
    # Truncate long texts to avoid token limits
    if len(text) > 8000:
        logger.warning(f"Text too long ({len(text)} chars), truncating to 8000 chars")
        text = text[:8000]
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Getting embedding for text of length {len(text)} (attempt {attempt+1}/{max_retries})")
            
            response = openai.embeddings.create(
                model=model,
                input=text
            )
            
            # Log the successful response
            if attempt > 0:
                logger.info(f"Successfully got embedding after {attempt+1} attempts")
                
            return response.data[0].embedding
            
        except Exception as e:
            error_msg = str(e)
            logger.warning(f"Error getting embedding (attempt {attempt+1}/{max_retries}): {error_msg}")
            
            # Check for specific error types
            if "rate limit" in error_msg.lower():
                # Rate limit error - use longer backoff
                wait_time = retry_delay * (5**attempt) 
                logger.info(f"Rate limit exceeded, waiting {wait_time} seconds")
                time.sleep(wait_time)
            elif "token" in error_msg.lower() and "exceed" in error_msg.lower():
                # Token limit exceeded, try with shorter text
                new_length = int(len(text) * 0.75)
                logger.info(f"Token limit exceeded, shortening text from {len(text)} to {new_length} chars")
                text = text[:new_length]
                time.sleep(retry_delay)
            else:
                # Other error, use standard backoff
                time.sleep(retry_delay * (2**attempt))
                
            if attempt == max_retries - 1:
                logger.error(f"Failed to get embedding after {max_retries} attempts")
                return None

def chunk_text(text, chunk_size):
    """Split text into chunks of approximately chunk_size characters."""
    if len(text) <= chunk_size:
        return [text]
    
    logger.info(f"Chunking text of length {len(text)} into ~{chunk_size} char chunks")
    
    # For JSON data, try to split at logical boundaries
    chunks = []
    current_chunk = ""
    bracket_count = 0
    brace_count = 0
    
    # Process character by character to handle JSON structure
    for char in text:
        current_chunk += char
        
        # Track JSON nesting
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
        elif char == '[':
            bracket_count += 1
        elif char == ']':
            bracket_count -= 1
        
        # Check if we're at a good break point
        if len(current_chunk) >= chunk_size:
            # Only break at a logical boundary - end of an object or array
            if (char in ']},' and brace_count <= 0 and bracket_count <= 0):
                chunks.append(current_chunk)
                current_chunk = ""
    
    # Add any remaining text
    if current_chunk:
        chunks.append(current_chunk)
    
    logger.info(f"Split text into {len(chunks)} chunks")
    
    # Log chunk sizes for debugging
    for i, chunk in enumerate(chunks):
        logger.info(f"Chunk {i}: {len(chunk)} chars")
    
    return chunks

def process_json_file(file_path, chunk_size):
    """Process a JSON file and generate embeddings for its content."""
    embeddings = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract filename without extension
        file_name = file_path.stem
        
        # Convert JSON to text representation
        json_text = json.dumps(data, indent=2)
        
        # Split into chunks if needed
        chunks = chunk_text(json_text, chunk_size)
        
        # Generate embeddings for each chunk
        for i, chunk in enumerate(chunks):
            # Create a unique ID for this chunk
            chunk_id = f"{file_name}_chunk_{i}"
            
            # Get embedding for the chunk
            embedding = get_embedding(chunk)
            
            if embedding:
                embeddings.append({
                    'id': chunk_id,
                    'text': chunk,
                    'embedding': embedding,
                    'metadata': {
                        'file_name': file_name,
                        'chunk_index': i,
                        'total_chunks': len(chunks)
                    }
                })
                
                # Small delay to avoid rate limits
                time.sleep(0.1)
        
        logger.info(f"Processed {file_name}: {len(embeddings)} chunks")
        return embeddings
    
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        return []

def initialize_pinecone(api_key, environment, index_name):
    """Initialize Pinecone client and connect to existing index or create a new one."""
    try:
        # Initialize Pinecone with new API
        pc = pinecone.Pinecone(api_key=api_key)
        
        # Try to connect directly to the known index host
        host = "virginia-data-portal-nprks3k.svc.aped-4627-b74a.pinecone.io"
        logger.info(f"Connecting directly to Pinecone index at: {host}")
        
        try:
            # Connect to the index directly using the host URL
            index = pc.Index(host=host)
            # Test the connection with a simple stats request
            stats = index.describe_index_stats()
            logger.info(f"Successfully connected to Pinecone index: {stats}")
            return index
        except Exception as direct_err:
            logger.error(f"Error connecting directly to index: {direct_err}")
            
            # Fall back to listing indexes and finding the right one
            logger.info("Falling back to listing indexes...")
            existing_indexes = pc.list_indexes()
            
            for idx in existing_indexes:
                if idx.name == index_name:
                    logger.info(f"Found index in list: {idx.name} at {idx.host}")
                    index = pc.Index(host=idx.host)
                    return index
            
            logger.error(f"Could not find index {index_name} in available indexes")
            return None
                
    except Exception as e:
        logger.error(f"Error initializing Pinecone: {str(e)}")
        return None

def store_embeddings_in_pinecone(index, embeddings, batch_size=100):
    """Store embeddings in Pinecone index using batching."""
    if index is None:
        logger.error("Cannot store embeddings: Pinecone index is None")
        return
        
    total_embeddings = len(embeddings)
    logger.info(f"Storing {total_embeddings} embeddings in Pinecone")
    
    successful_vectors = 0
    
    # Process in batches
    for i in range(0, total_embeddings, batch_size):
        batch = embeddings[i:i+batch_size]
        
        # Convert to the Pinecone format
        vectors = []
        for item in batch:
            vectors.append({
                "id": item['id'],
                "values": item['embedding'],
                "metadata": {
                    'text': item['text'],
                    **item['metadata']
                }
            })
        
        try:
            # Log the first vector to check format is correct
            if i == 0:
                logger.info(f"Sample vector format: {vectors[0]}")
            
            # Upsert batch using standard upsert method
            upsert_response = index.upsert(vectors=vectors)
            logger.info(f"Stored batch {i//batch_size + 1}/{(total_embeddings+batch_size-1)//batch_size}")
            logger.info(f"Upsert response: {upsert_response}")
            
            successful_vectors += len(vectors)
        except Exception as e:
            logger.error(f"Error upserting batch to Pinecone: {str(e)}")
            # Try individual vectors to see which ones might be causing issues
            for v_idx, vector in enumerate(vectors):
                try:
                    single_response = index.upsert(vectors=[vector])
                    logger.info(f"Stored single vector {v_idx} with id {vector['id']}")
                    successful_vectors += 1
                except Exception as single_err:
                    logger.error(f"Error upserting single vector {v_idx} with id {vector['id']}: {str(single_err)}")
        
        # Sleep to avoid rate limits
        time.sleep(0.5)
    
    logger.info(f"Successfully stored {successful_vectors} out of {total_embeddings} vectors")

def main():
    """Main function to process JSON files, generate embeddings and store in Pinecone."""
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
    
    # Test OpenAI connection
    try:
        test_embedding = get_embedding("Test connection to OpenAI API")
        if test_embedding:
            logger.info("Successfully connected to OpenAI API")
        else:
            logger.error("Failed to get test embedding from OpenAI")
            return
    except Exception as e:
        logger.error(f"Failed to connect to OpenAI: {str(e)}")
        return
    
    # Initialize Pinecone
    index = initialize_pinecone(
        api_key=pinecone_api_key,
        environment=args.pinecone_env,
        index_name=args.index_name
    )
    
    if index is None:
        logger.error("Failed to initialize Pinecone. Exiting.")
        return
    
    # Test Pinecone connection with a simple upsert
    try:
        test_vector = {
            "id": "test_vector",
            "values": [0.1] * EMBEDDING_DIMENSION,
            "metadata": {"text": "Test vector", "test": True}
        }
        upsert_response = index.upsert(vectors=[test_vector])
        logger.info(f"Test upsert response: {upsert_response}")
        
        # Verify the test vector exists in the index
        query_response = index.query(vector=[0.1] * EMBEDDING_DIMENSION, top_k=1, include_metadata=True)
        logger.info(f"Test query response: {query_response}")
        
        if query_response and query_response.matches:
            logger.info("Successfully verified Pinecone connection with test vector")
        else:
            logger.warning("Could not verify test vector in Pinecone")
    except Exception as e:
        logger.error(f"Error testing Pinecone connection: {str(e)}")
        logger.warning("Continuing despite test failure...")
    
    # Get all JSON files in the directory
    json_files = list(json_path.glob("*.json"))
    total_files = len(json_files)
    logger.info(f"Found {total_files} JSON files to process")
    
    # Limit number of files if specified
    if args.num_files is not None:
        json_files = json_files[:args.num_files]
        logger.info(f"Processing {len(json_files)} files out of {total_files}")
    
    # Process all JSON files
    all_embeddings = []
    
    for file_path in tqdm(json_files, desc="Processing JSON files"):
        file_embeddings = process_json_file(file_path, args.chunk_size)
        all_embeddings.extend(file_embeddings)
    
    logger.info(f"Generated {len(all_embeddings)} embeddings in total")
    
    # Store embeddings in Pinecone
    store_embeddings_in_pinecone(index, all_embeddings, batch_size=args.batch_size)
    
    try:
        # Print stats from Pinecone
        stats = index.describe_index_stats()
        logger.info(f"Pinecone index stats: {stats}")
        
        print("\n=== Embedding and Storage Complete ===")
        print(f"Total embeddings created and stored: {len(all_embeddings)}")
        print(f"Pinecone index: {args.index_name}")
        print(f"Vector count in index: {stats['total_vector_count']}")
    except Exception as e:
        logger.error(f"Error getting index stats: {str(e)}")
        print("\n=== Embedding and Storage Complete with Errors ===")
        print(f"Total embeddings created: {len(all_embeddings)}")
        print("Failed to get Pinecone index stats")

if __name__ == "__main__":
    main() 