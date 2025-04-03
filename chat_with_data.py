#!/usr/bin/env python3
"""
ChatGPT Integration for Virginia Data Portal JSON Data

This script:
1. Connects to the Pinecone index containing embedded JSON data
2. Allows interactive querying using ChatGPT
3. Provides context-aware responses based on the embedded data
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
        logging.FileHandler("chat_with_data.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("chat_with_data")

# Default models
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-3.5-turbo"

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Chat with embedded JSON data using ChatGPT')
    parser.add_argument('--api-key', required=False, help='OpenAI API key')
    parser.add_argument('--pinecone-key', required=False, help='Pinecone API key')
    parser.add_argument('--pinecone-env', default='us-east-1-aws', help='Pinecone environment')
    parser.add_argument('--index-name', default='virginia-data-portal', help='Pinecone index name')
    parser.add_argument('--top-k', type=int, default=3, help='Number of results to retrieve for context')
    parser.add_argument('--chat-model', default=CHAT_MODEL, help='OpenAI chat model to use')
    return parser.parse_args()

def get_embedding(text, model=EMBEDDING_MODEL):
    """Get embedding from OpenAI API."""
    try:
        response = openai.embeddings.create(
            model=model,
            input=text
        )
        return response.data[0].embedding
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

def search_json_data(index, query, top_k=3):
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
                'text': match.metadata.get('text', '')
            })
        
        return processed_results
    
    except Exception as e:
        logger.error(f"Error searching Pinecone: {str(e)}")
        return []

def format_context_for_chatgpt(results):
    """Format search results as context for ChatGPT."""
    if not results:
        return "No relevant data found."
    
    context = "Here is some relevant data from the Virginia Data Portal:\n\n"
    
    for i, result in enumerate(results, 1):
        context += f"Data from file '{result['file_name']}' (relevance: {result['score']:.4f}):\n"
        context += f"{result['text']}\n\n"
    
    return context

def chat_with_data(index, query, chat_model=CHAT_MODEL, top_k=3):
    """Chat with the embedded data using ChatGPT."""
    # Search for relevant data
    results = search_json_data(index, query, top_k=top_k)
    
    # Format context for ChatGPT
    context = format_context_for_chatgpt(results)
    
    # Create messages for ChatGPT
    messages = [
        {"role": "system", "content": "You are a helpful assistant that provides information about Virginia data. Use the provided context to answer questions accurately. If the context doesn't contain enough information to answer the question, say so."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]
    
    # Get response from ChatGPT
    try:
        response = openai.chat.completions.create(
            model=chat_model,
            messages=messages,
            temperature=0.3,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        logger.error(f"Error getting response from ChatGPT: {str(e)}")
        return f"Error: {str(e)}"

def main():
    """Main function to chat with embedded data."""
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
    
    print("\n=== Virginia Data Portal Chat ===")
    print("Type 'exit' or 'quit' to end the conversation.")
    print("Ask questions about the Virginia data to get information.\n")
    
    while True:
        query = input("\nYour question: ")
        
        if query.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        
        if not query.strip():
            continue
        
        print("\nSearching for relevant data...")
        response = chat_with_data(
            index=index,
            query=query,
            chat_model=args.chat_model,
            top_k=args.top_k
        )
        
        print("\nAnswer:")
        print(response)

if __name__ == "__main__":
    main() 