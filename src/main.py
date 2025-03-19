"""
Main entry point for Virginia Open Data Portal Metadata Extractor.
"""
import argparse
import os
from pathlib import Path
import sys
import json
from typing import List, Dict, Any

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.metadata_extractor import VirginiaDataPortalExtractor
from src.api.endpoints import start_server
from src.utils.crawler import VirginiaPortalCrawler

def main():
    """Main function to run the application."""
    parser = argparse.ArgumentParser(description="Virginia Open Data Portal Metadata Extractor")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Extract command
    extract_parser = subparsers.add_parser("extract", help="Extract metadata from Virginia Open Data Portal")
    
    # API command
    api_parser = subparsers.add_parser("api", help="Start the API server")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search for datasets")
    search_parser.add_argument("query", help="Query string to search for")
    search_parser.add_argument("--limit", type=int, default=5, help="Maximum number of results to return")
    
    # Crawl command
    crawl_parser = subparsers.add_parser("crawl", help="Crawl the Virginia Open Data Portal website")
    crawl_parser.add_argument("--output", type=str, default="./data/crawled", 
                            help="Directory to save crawled metadata")
    crawl_parser.add_argument("--start-page", type=int, default=1, 
                            help="Page number to start crawling from")
    crawl_parser.add_argument("--end-page", type=int, default=10, 
                            help="Page number to end crawling at")
    
    # Embed crawled data command
    embed_parser = subparsers.add_parser("embed-crawled", 
                                       help="Generate embeddings for crawled metadata")
    embed_parser.add_argument("--input", type=str, default="./data/crawled", 
                            help="Directory containing crawled metadata")
    
    args = parser.parse_args()
    
    if args.command == "extract":
        extractor = VirginiaDataPortalExtractor()
        extractor.run_pipeline()
    elif args.command == "api":
        start_server()
    elif args.command == "search":
        extractor = VirginiaDataPortalExtractor()
        results = extractor.search_similar_datasets(args.query, args.limit)
        print(f"Found {len(results['ids'])} results for query: {args.query}")
        for i in range(len(results['ids'])):
            print(f"\nResult {i+1}:")
            print(f"ID: {results['ids'][i]}")
            print(f"Title: {results['metadatas'][i]['title']}")
            print(f"Description: {results['metadatas'][i]['description'][:100]}...")
            print(f"Tags: {', '.join(results['metadatas'][i]['tags'][:5])}")
            print(f"URL: {results['metadatas'][i].get('source_page', '')}")
    elif args.command == "crawl":
        # Set environment variables from command-line arguments
        if args.start_page:
            os.environ['CRAWLER_START_PAGE'] = str(args.start_page)
        if args.end_page:
            os.environ['CRAWLER_END_PAGE'] = str(args.end_page)
        
        crawler = VirginiaPortalCrawler(output_dir=args.output)
        crawler.run()
    elif args.command == "embed-crawled":
        embed_crawled_data(args.input)
    else:
        parser.print_help()

def embed_crawled_data(input_dir: str):
    """Generate embeddings for crawled metadata and store in ChromaDB.
    
    Args:
        input_dir: Directory containing crawled metadata JSON files
    """
    print(f"Generating embeddings for crawled metadata in {input_dir}")
    
    # Load all datasets
    all_datasets_file = Path(input_dir) / "all_datasets.json"
    if not all_datasets_file.exists():
        print(f"Error: {all_datasets_file} not found. Run the crawler first.")
        return
    
    with open(all_datasets_file, 'r') as f:
        all_datasets = json.load(f)
    
    print(f"Found {len(all_datasets)} datasets")
    
    # Initialize the extractor
    extractor = VirginiaDataPortalExtractor()
    
    # Process each dataset
    for dataset in all_datasets:
        try:
            # Prepare metadata for embedding
            metadata = {
                'id': dataset['id'],
                'title': dataset['title'],
                'description': dataset['description'],
                'tags': dataset['tags'],
                'urls': [resource['url'] for resource in dataset.get('resources', [])],
                'resource_count': dataset.get('resource_count', 0),
                'organization': dataset.get('organization', ''),
                'source_page': dataset.get('url', '')
            }
            
            # Generate text for embedding
            text_for_embedding = f"{metadata['title']} {metadata['description']} {' '.join(metadata['tags'])}"
            
            # Generate embedding
            embedding = extractor.generate_embedding(text_for_embedding)
            
            # Store in ChromaDB
            extractor.collection.add(
                embeddings=[embedding],
                documents=[text_for_embedding],
                metadatas=[metadata],
                ids=[metadata['id']]
            )
            
            print(f"Processed dataset: {metadata['id']}")
        except Exception as e:
            print(f"Error processing dataset {dataset.get('id', 'unknown')}: {str(e)}")
    
    print("Embedding generation completed successfully!")

if __name__ == "__main__":
    main() 