#!/usr/bin/env python3
"""
Virginia Data Portal Search Frontend

A simple Flask application that provides a web interface to search the
Virginia Data Portal embeddings stored in Pinecone using semantic search.
"""

import os
import json
import logging
from pathlib import Path
from flask import Flask, request, render_template, jsonify
import openai
import pinecone

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("search_app")

# Default OpenAI model for embeddings
EMBEDDING_MODEL = "text-embedding-ada-002"

# Initialize Flask app
app = Flask(__name__, template_folder="templates", static_folder="static")

# Load configuration
@app.before_first_request
def initialize():
    # Get API keys from environment variables
    app.config['OPENAI_API_KEY'] = os.environ.get("OPENAI_API_KEY")
    app.config['PINECONE_API_KEY'] = os.environ.get("PINECONE_API_KEY")
    app.config['PINECONE_ENV'] = os.environ.get("PINECONE_ENV", "us-west1-gcp")
    app.config['PINECONE_INDEX'] = os.environ.get("PINECONE_INDEX", "virginia-data")
    
    if not app.config['OPENAI_API_KEY']:
        logger.error("OPENAI_API_KEY not set in environment variables")
    
    if not app.config['PINECONE_API_KEY']:
        logger.error("PINECONE_API_KEY not set in environment variables")
    
    # Initialize OpenAI
    openai.api_key = app.config['OPENAI_API_KEY']
    
    # Initialize Pinecone connection
    pinecone.init(
        api_key=app.config['PINECONE_API_KEY'],
        environment=app.config['PINECONE_ENV']
    )
    
    # Connect to Pinecone index
    try:
        app.config['index'] = pinecone.Index(app.config['PINECONE_INDEX'])
        stats = app.config['index'].describe_index_stats()
        logger.info(f"Connected to Pinecone index: {app.config['PINECONE_INDEX']}")
        logger.info(f"Index stats: {stats}")
    except Exception as e:
        logger.error(f"Error connecting to Pinecone: {str(e)}")
        app.config['index'] = None

def get_embedding(text, model=EMBEDDING_MODEL):
    """Get embedding from OpenAI API."""
    if not text:
        return None
    
    try:
        response = openai.Embedding.create(
            model=model,
            input=text
        )
        return response["data"][0]["embedding"]
    except Exception as e:
        logger.error(f"Error getting embedding: {str(e)}")
        return None

def search_pinecone(query_embedding, filter_type=None, top_k=10):
    """Search Pinecone index with the query embedding."""
    if not app.config.get('index'):
        logger.error("Pinecone index not initialized")
        return []
    
    # Set up metadata filter if needed
    filter_dict = {}
    if filter_type:
        filter_dict = {"type": filter_type}
    
    # Query Pinecone
    try:
        results = app.config['index'].query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict
        )
        return results
    except Exception as e:
        logger.error(f"Error querying Pinecone: {str(e)}")
        return None

@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    """Handle search requests."""
    query = request.form.get('query', '')
    filter_type = request.form.get('filter', None)  # 'dataset' or 'column'
    
    if not query:
        return jsonify({"error": "Please provide a search query"})
    
    # Get embedding for query
    query_embedding = get_embedding(query)
    if not query_embedding:
        return jsonify({"error": "Failed to generate embedding for query"})
    
    # Search Pinecone
    results = search_pinecone(query_embedding, filter_type)
    if not results:
        return jsonify({"error": "Search failed or no results found"})
    
    # Format results for display
    formatted_results = []
    for match in results.get('matches', []):
        # Extract metadata
        metadata = match.get('metadata', {})
        result_type = metadata.get('type', '')
        dataset_id = metadata.get('dataset_id', '')
        dataset_title = metadata.get('dataset_title', '')
        
        # Common fields
        result = {
            'id': match.get('id', ''),
            'score': match.get('score', 0),
            'type': result_type,
            'dataset_id': dataset_id,
            'dataset_title': dataset_title
        }
        
        # Add type-specific fields
        if result_type == 'dataset':
            result['title'] = metadata.get('title', '')
            result['organization'] = metadata.get('organization', '')
            result['tags'] = metadata.get('tags', [])
        elif result_type == 'column':
            result['column_name'] = metadata.get('column_name', '')
            result['column_type'] = metadata.get('column_type', '')
        
        # Add raw text for context
        result['text'] = metadata.get('text', '')
        
        formatted_results.append(result)
    
    return jsonify({
        "query": query,
        "filter": filter_type,
        "results": formatted_results
    })

@app.route('/dataset/<dataset_id>')
def dataset_detail(dataset_id):
    """Show details for a specific dataset."""
    # This would typically fetch the dataset details from a database
    # For now, we'll return a placeholder
    return jsonify({"message": f"Details for dataset {dataset_id} not implemented yet"})

if __name__ == '__main__':
    # Create templates and static directories
    templates_dir = Path(__file__).resolve().parent / "templates"
    static_dir = Path(__file__).resolve().parent / "static"
    
    templates_dir.mkdir(exist_ok=True)
    static_dir.mkdir(exist_ok=True)
    
    # Run the app
    app.run(host='0.0.0.0', port=5000, debug=True) 