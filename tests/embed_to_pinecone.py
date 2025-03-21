"""
Pinecone Embedding Tool

This script processes crawled datasets (JSON and Markdown files) from the Virginia Open Data Portal,
generates text embeddings using OpenAI's API, and stores them in a Pinecone vector database
for semantic search capabilities.

Usage:
    python embed_to_pinecone.py [--data-dir PATH] [--index-name NAME]

Requirements:
    - OpenAI API key set as environment variable OPENAI_API_KEY
    - Pinecone API key set as environment variable PINECONE_API_KEY
    - Pinecone environment set as environment variable PINECONE_ENVIRONMENT (optional)
"""

import os
import json
import glob
import time
import logging
import argparse
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import openai
import pinecone
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("embedding.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("embedding")

# Configuration
DEFAULT_DATA_DIR = "data/crawled_datasets"
DEFAULT_INDEX_NAME = "virginia-data"
DEFAULT_BATCH_SIZE = 100
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_EMBEDDING_DIMENSIONS = 1536  # For text-embedding-3-small
DEFAULT_NAMESPACE = "virginia-datasets"

class PineconeEmbedder:
    def __init__(
        self, 
        data_dir: str = DEFAULT_DATA_DIR,
        index_name: str = DEFAULT_INDEX_NAME,
        batch_size: int = DEFAULT_BATCH_SIZE,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        namespace: str = DEFAULT_NAMESPACE
    ):
        """Initialize the embedder with configuration."""
        self.data_dir = Path(data_dir)
        self.index_name = index_name
        self.batch_size = batch_size
        self.embedding_model = embedding_model
        self.namespace = namespace
        
        # Check if data directory exists
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        # Initialize OpenAI client
        self._init_openai()
        
        # Initialize Pinecone
        self._init_pinecone()
        
        # Track metrics
        self.stats = {
            "datasets_processed": 0,
            "documents_embedded": 0,
            "tokens_processed": 0,
            "api_calls": 0,
            "chunks_created": 0,
        }
        
        logger.info(f"Initialized embedder for data in: {self.data_dir}")
        logger.info(f"Using Pinecone index: {self.index_name}")
        logger.info(f"Using embedding model: {self.embedding_model}")
    
    def _init_openai(self):
        """Initialize OpenAI API client."""
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        openai.api_key = api_key
        logger.info("OpenAI API initialized")
    
    def _init_pinecone(self):
        """Initialize Pinecone connection and ensure index exists."""
        api_key = os.environ.get("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("Pinecone API key not found. Set PINECONE_API_KEY environment variable.")
            
        # Get environment or use default
        environment = os.environ.get("PINECONE_ENVIRONMENT", "gcp-starter")
        
        # Initialize Pinecone
        pinecone.init(api_key=api_key, environment=environment)
        
        # Check if index exists, create if not
        if self.index_name not in pinecone.list_indexes():
            logger.info(f"Creating new Pinecone index: {self.index_name}")
            pinecone.create_index(
                name=self.index_name,
                dimension=DEFAULT_EMBEDDING_DIMENSIONS,
                metric="cosine",
                spec={"serverless": {"cloud": "aws", "region": "us-west-2"}}
            )
        
        # Connect to index
        self.index = pinecone.Index(self.index_name)
        stats = self.index.describe_index_stats()
        logger.info(f"Connected to Pinecone index: {self.index_name}")
        logger.info(f"Index stats: {stats}")
    
    def find_dataset_files(self) -> List[Dict[str, str]]:
        """Find all JSON and corresponding Markdown files in the data directory."""
        dataset_files = []
        
        # Find all JSON files that aren't summary files
        json_files = list(self.data_dir.glob("*.json"))
        json_files = [f for f in json_files if not f.name.startswith("summary_") and not f.name == "crawl_summary.json"]
        
        for json_file in json_files:
            dataset_id = json_file.stem
            md_file = self.data_dir / f"{dataset_id}.md"
            
            if md_file.exists():
                dataset_files.append({
                    "id": dataset_id,
                    "json_path": str(json_file),
                    "md_path": str(md_file),
                })
        
        logger.info(f"Found {len(dataset_files)} dataset files to process")
        return dataset_files
    
    def read_dataset(self, dataset_file: Dict[str, str]) -> Dict[str, Any]:
        """Read a dataset from its JSON and MD files."""
        # Read JSON metadata
        with open(dataset_file["json_path"], "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        # Read markdown content
        with open(dataset_file["md_path"], "r", encoding="utf-8") as f:
            markdown_content = f.read()
        
        return {
            "id": dataset_file["id"],
            "metadata": metadata,
            "markdown": markdown_content,
        }
    
    def chunk_markdown(self, markdown: str, max_tokens: int = 1000) -> List[str]:
        """Split markdown content into chunks, respecting section boundaries where possible."""
        # Split by markdown headers
        sections = []
        current_section = []
        
        for line in markdown.split("\n"):
            if line.startswith("# ") or line.startswith("## ") or line.startswith("### "):
                if current_section:
                    sections.append("\n".join(current_section))
                current_section = [line]
            else:
                current_section.append(line)
        
        # Add the last section
        if current_section:
            sections.append("\n".join(current_section))
        
        # Now split sections that are too large
        chunks = []
        for section in sections:
            # Rough token count estimation (4 chars ~= 1 token)
            estimated_tokens = len(section) // 4
            
            if estimated_tokens <= max_tokens:
                chunks.append(section)
            else:
                # Split by paragraphs
                paragraphs = section.split("\n\n")
                current_chunk = []
                current_token_count = 0
                
                for para in paragraphs:
                    para_tokens = len(para) // 4
                    
                    if current_token_count + para_tokens > max_tokens:
                        if current_chunk:
                            chunks.append("\n\n".join(current_chunk))
                        current_chunk = [para]
                        current_token_count = para_tokens
                    else:
                        current_chunk.append(para)
                        current_token_count += para_tokens
                
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
        
        self.stats["chunks_created"] += len(chunks)
        return chunks
    
    def generate_embeddings(self, texts: List[str]) -> Tuple[List[List[float]], int]:
        """Generate embeddings for a list of texts using OpenAI's API."""
        try:
            response = openai.Embedding.create(
                model=self.embedding_model,
                input=texts
            )
            self.stats["api_calls"] += 1
            
            embeddings = [item["embedding"] for item in response["data"]]
            
            # Update token usage stats
            if "usage" in response:
                self.stats["tokens_processed"] += response["usage"]["total_tokens"]
                token_count = response["usage"]["total_tokens"]
            else:
                # If no usage info, estimate
                token_count = sum(len(text) // 4 for text in texts)
                self.stats["tokens_processed"] += token_count
                
            return embeddings, token_count
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            # In case of error, return empty embeddings
            return [], 0
    
    def prepare_document_vectors(self, dataset: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Prepare vectors for a dataset by chunking and embedding markdown content."""
        dataset_id = dataset["id"]
        metadata = dataset["metadata"]
        markdown = dataset["markdown"]
        
        # Create a stable prefix for vector IDs
        id_prefix = f"dataset_{dataset_id}"
        
        # Basic metadata to include with each vector
        base_metadata = {
            "dataset_id": dataset_id,
            "title": metadata["title"],
            "organization": metadata.get("organization", ""),
            "tags": metadata.get("tags", []),
            "url": metadata.get("url", ""),
        }
        
        # Chunk the markdown
        chunks = self.chunk_markdown(markdown)
        logger.info(f"Split dataset '{metadata['title']}' into {len(chunks)} chunks")
        
        # Prepare document records without embeddings yet
        documents = []
        for i, chunk in enumerate(chunks):
            # Create a unique ID for this chunk
            chunk_id = f"{id_prefix}_chunk_{i}"
            
            # Extract section title if present (first line if it's a header)
            first_line = chunk.split("\n")[0].strip()
            section_title = first_line if first_line.startswith("#") else ""
            
            # Chunk-specific metadata
            chunk_metadata = base_metadata.copy()
            chunk_metadata.update({
                "chunk_id": i,
                "section_title": section_title.replace("#", "").strip(),
                "chunk_text": chunk,
                "total_chunks": len(chunks),
            })
            
            documents.append({
                "id": chunk_id,
                "text": chunk,
                "metadata": chunk_metadata
            })
        
        return documents
    
    def upsert_vectors_to_pinecone(self, documents_with_embeddings: List[Dict[str, Any]]):
        """Upload vectors to Pinecone in batches."""
        vectors = []
        
        for doc in documents_with_embeddings:
            vectors.append({
                "id": doc["id"],
                "values": doc["embedding"],
                "metadata": doc["metadata"]
            })
        
        # Insert in batches
        batch_size = self.batch_size
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i+batch_size]
            try:
                self.index.upsert(vectors=batch, namespace=self.namespace)
                logger.info(f"Uploaded batch of {len(batch)} vectors to Pinecone")
            except Exception as e:
                logger.error(f"Error uploading vectors to Pinecone: {str(e)}")
    
    def process_dataset(self, dataset_file: Dict[str, str]):
        """Process a single dataset file, embed it, and upload to Pinecone."""
        try:
            # Read the dataset
            dataset = self.read_dataset(dataset_file)
            logger.info(f"Processing dataset: {dataset['metadata']['title']}")
            
            # Prepare documents (chunk the text)
            documents = self.prepare_document_vectors(dataset)
            
            # Process in batches for embedding
            batch_size = 20  # Smaller batch size for API calls
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i+batch_size]
                
                # Extract text for embedding
                texts = [doc["text"] for doc in batch]
                
                # Generate embeddings
                embeddings, tokens = self.generate_embeddings(texts)
                
                if not embeddings:
                    logger.warning(f"No embeddings generated for batch in dataset {dataset['id']}")
                    continue
                
                # Add embeddings to documents
                for j, embedding in enumerate(embeddings):
                    batch[j]["embedding"] = embedding
                
                # Upload to Pinecone
                self.upsert_vectors_to_pinecone(batch)
                
                # Rate limiting for OpenAI API
                time.sleep(0.5)
            
            # Update stats
            self.stats["datasets_processed"] += 1
            self.stats["documents_embedded"] += len(documents)
            
            logger.info(f"Successfully processed dataset: {dataset['metadata']['title']}")
            logger.info(f"Created {len(documents)} vector entries")
            
        except Exception as e:
            logger.error(f"Error processing dataset {dataset_file['id']}: {str(e)}")
    
    def process_all_datasets(self):
        """Process all datasets in the data directory."""
        # Find all dataset files
        dataset_files = self.find_dataset_files()
        
        # Process each dataset
        for dataset_file in tqdm(dataset_files, desc="Processing datasets"):
            self.process_dataset(dataset_file)
        
        # Print final stats
        logger.info("\n=== Embedding Process Complete ===")
        logger.info(f"Datasets processed: {self.stats['datasets_processed']}")
        logger.info(f"Documents embedded: {self.stats['documents_embedded']}")
        logger.info(f"Chunks created: {self.stats['chunks_created']}")
        logger.info(f"Tokens processed: {self.stats['tokens_processed']}")
        logger.info(f"API calls made: {self.stats['api_calls']}")
        
        # Get index stats
        stats = self.index.describe_index_stats()
        logger.info(f"Final index stats: {stats}")

def main():
    """Main entry point for embedding datasets to Pinecone."""
    parser = argparse.ArgumentParser(
        description="Process datasets from the Virginia Open Data Portal and upload embeddings to Pinecone"
    )
    
    parser.add_argument(
        "--data-dir", 
        default=DEFAULT_DATA_DIR,
        help=f"Directory containing dataset files (default: {DEFAULT_DATA_DIR})"
    )
    
    parser.add_argument(
        "--index-name", 
        default=DEFAULT_INDEX_NAME,
        help=f"Name of the Pinecone index (default: {DEFAULT_INDEX_NAME})"
    )
    
    parser.add_argument(
        "--batch-size", 
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for Pinecone uploads (default: {DEFAULT_BATCH_SIZE})"
    )
    
    parser.add_argument(
        "--embedding-model", 
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"OpenAI embedding model to use (default: {DEFAULT_EMBEDDING_MODEL})"
    )
    
    parser.add_argument(
        "--namespace", 
        default=DEFAULT_NAMESPACE,
        help=f"Pinecone namespace to use (default: {DEFAULT_NAMESPACE})"
    )
    
    args = parser.parse_args()
    
    # Print banner
    print("\n=== Virginia Data Embedder for Pinecone ===")
    print(f"Data directory: {args.data_dir}")
    print(f"Pinecone index: {args.index_name}")
    print(f"Embedding model: {args.embedding_model}")
    print(f"Namespace: {args.namespace}")
    
    # Check for API keys
    if not os.environ.get("OPENAI_API_KEY"):
        print("\nWARNING: OPENAI_API_KEY environment variable not set.")
        print("You can set it in a .env file or use: export OPENAI_API_KEY=your_key_here")
        return
    
    if not os.environ.get("PINECONE_API_KEY"):
        print("\nWARNING: PINECONE_API_KEY environment variable not set.")
        print("You can set it in a .env file or use: export PINECONE_API_KEY=your_key_here")
        return
    
    # Create and run embedder
    try:
        embedder = PineconeEmbedder(
            data_dir=args.data_dir,
            index_name=args.index_name,
            batch_size=args.batch_size,
            embedding_model=args.embedding_model,
            namespace=args.namespace
        )
        
        embedder.process_all_datasets()
        
        print("\n=== Embedding Complete ===")
        print(f"Datasets processed: {embedder.stats['datasets_processed']}")
        print(f"Documents embedded: {embedder.stats['documents_embedded']}")
        print(f"Chunks created: {embedder.stats['chunks_created']}")
        print(f"Tokens processed: {embedder.stats['tokens_processed']}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    main() 