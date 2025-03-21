"""
Pinecone Vector Search with ChatGPT

This script enables semantic search over your Virginia Open Data Portal datasets
that have been embedded and stored in Pinecone. It uses OpenAI to generate
embeddings for search queries and ChatGPT to provide natural language responses
about the search results.

Usage:
    python search_pinecone.py [--index-name NAME] [--model MODEL]

Requirements:
    - OpenAI API key set as environment variable OPENAI_API_KEY
    - Pinecone API key set as environment variable PINECONE_API_KEY
    - Pinecone environment set as environment variable PINECONE_ENVIRONMENT (optional)
"""

import os
import json
import time
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import openai
import pinecone
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

# Load environment variables from .env file if present
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("search.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("search")

# Configuration
DEFAULT_INDEX_NAME = "virginia-data"
DEFAULT_NAMESPACE = "virginia-datasets"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_CHAT_MODEL = "gpt-3.5-turbo"
DEFAULT_TOP_K = 5

# Rich console for pretty output
console = Console()

class PineconeSearcher:
    def __init__(
        self,
        index_name: str = DEFAULT_INDEX_NAME,
        namespace: str = DEFAULT_NAMESPACE,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        chat_model: str = DEFAULT_CHAT_MODEL,
        top_k: int = DEFAULT_TOP_K
    ):
        """Initialize the Pinecone searcher."""
        self.index_name = index_name
        self.namespace = namespace
        self.embedding_model = embedding_model
        self.chat_model = chat_model
        self.top_k = top_k
        
        # Initialize API clients
        self._init_openai()
        self._init_pinecone()
        
        logger.info(f"Initialized searcher with Pinecone index: {self.index_name}")
        logger.info(f"Using embedding model: {self.embedding_model}")
        logger.info(f"Using chat model: {self.chat_model}")
    
    def _init_openai(self):
        """Initialize OpenAI API client."""
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        openai.api_key = api_key
        logger.info("OpenAI API initialized")
    
    def _init_pinecone(self):
        """Initialize Pinecone connection."""
        api_key = os.environ.get("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("Pinecone API key not found. Set PINECONE_API_KEY environment variable.")
            
        # Get environment or use default
        environment = os.environ.get("PINECONE_ENVIRONMENT", "gcp-starter")
        
        # Initialize Pinecone
        pinecone.init(api_key=api_key, environment=environment)
        
        # Check if index exists
        if self.index_name not in pinecone.list_indexes():
            raise ValueError(f"Pinecone index '{self.index_name}' not found. Create it first with embed_to_pinecone.py")
        
        # Connect to index
        self.index = pinecone.Index(self.index_name)
        stats = self.index.describe_index_stats()
        logger.info(f"Connected to Pinecone index: {self.index_name}")
        logger.info(f"Index stats: {stats}")
        
        # Check if vectors exist in the namespace
        if stats["namespaces"].get(self.namespace, {}).get("vector_count", 0) == 0:
            raise ValueError(f"No vectors found in namespace '{self.namespace}'. Run embed_to_pinecone.py first.")
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate an embedding for a single text using OpenAI's API."""
        try:
            response = openai.Embedding.create(
                model=self.embedding_model,
                input=text
            )
            
            embedding = response["data"][0]["embedding"]
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise
    
    def search_pinecone(self, query_embedding: List[float]) -> List[Dict[str, Any]]:
        """Search Pinecone for similar vectors."""
        try:
            results = self.index.query(
                vector=query_embedding,
                top_k=self.top_k,
                include_metadata=True,
                namespace=self.namespace
            )
            
            return results.get("matches", [])
            
        except Exception as e:
            logger.error(f"Error searching Pinecone: {str(e)}")
            raise
    
    def format_search_results(self, results: List[Dict[str, Any]]) -> str:
        """Format search results as a structured string for ChatGPT."""
        if not results:
            return "No matching datasets found."
        
        formatted_results = "### SEARCH RESULTS\n\n"
        
        for i, match in enumerate(results):
            metadata = match["metadata"]
            score = match["score"]
            
            formatted_results += f"#### Result {i+1} (Similarity: {score:.4f})\n\n"
            formatted_results += f"**Dataset**: {metadata.get('title', 'Untitled')}\n"
            formatted_results += f"**ID**: {metadata.get('dataset_id', 'Unknown')}\n"
            formatted_results += f"**Organization**: {metadata.get('organization', 'Unknown')}\n"
            
            if metadata.get("tags"):
                if isinstance(metadata["tags"], list):
                    formatted_results += f"**Tags**: {', '.join(metadata['tags'])}\n"
                else:
                    formatted_results += f"**Tags**: {metadata['tags']}\n"
            
            if metadata.get("section_title"):
                formatted_results += f"**Section**: {metadata['section_title']}\n"
            
            formatted_results += f"**URL**: {metadata.get('url', 'No URL')}\n\n"
            
            # Include the actual text content that matched
            if metadata.get("chunk_text"):
                chunk_text = metadata["chunk_text"]
                # Truncate if too long
                if len(chunk_text) > 1500:
                    chunk_text = chunk_text[:1500] + "..."
                formatted_results += f"**Content**:\n\n{chunk_text}\n\n"
            
            formatted_results += "---\n\n"
        
        return formatted_results
    
    def generate_chat_response(self, query: str, formatted_results: str) -> str:
        """Generate a ChatGPT response to the query based on search results."""
        system_prompt = """You are an assistant that helps users find information about datasets from the Virginia Open Data Portal.
Your task is to analyze the search results provided and answer the user's query based on that information.
Be conversational but concise. Focus on answering the question directly based on the search results.
If the search results don't contain relevant information to answer the query, acknowledge that and suggest how the user might refine their search.
Include specific details from the datasets when available and relevant.
"""

        user_prompt = f"""Here is my query: {query}

Here are the search results from the Virginia Open Data Portal:

{formatted_results}

Please provide a helpful response based on these search results."""

        try:
            response = openai.ChatCompletion.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.5,
                max_tokens=1000
            )
            
            return response["choices"][0]["message"]["content"]
            
        except Exception as e:
            logger.error(f"Error generating chat response: {str(e)}")
            return f"Error generating response: {str(e)}"
    
    def display_results_table(self, results: List[Dict[str, Any]]):
        """Display search results in a formatted table."""
        if not results:
            console.print(Panel("No matching datasets found.", title="Search Results", border_style="red"))
            return
        
        table = Table(title="Search Results", expand=True)
        table.add_column("Score", justify="center", style="cyan", no_wrap=True)
        table.add_column("Dataset", style="green")
        table.add_column("Organization", style="yellow")
        table.add_column("Section", style="magenta")
        
        for match in results:
            metadata = match["metadata"]
            score = match["score"]
            
            title = metadata.get("title", "Untitled")
            org = metadata.get("organization", "Unknown")
            section = metadata.get("section_title", "")
            
            table.add_row(
                f"{score:.4f}",
                title,
                org,
                section
            )
        
        console.print(table)
    
    def display_full_result(self, result: Dict[str, Any]):
        """Display full details for a single search result."""
        metadata = result["metadata"]
        score = result["score"]
        
        # Create a panel with the result details
        title = f"Dataset: {metadata.get('title', 'Untitled')} (Score: {score:.4f})"
        
        content = []
        content.append(f"**ID**: {metadata.get('dataset_id', 'Unknown')}")
        content.append(f"**Organization**: {metadata.get('organization', 'Unknown')}")
        
        if metadata.get("tags"):
            if isinstance(metadata["tags"], list):
                content.append(f"**Tags**: {', '.join(metadata['tags'])}")
            else:
                content.append(f"**Tags**: {metadata['tags']}")
        
        if metadata.get("section_title"):
            content.append(f"**Section**: {metadata['section_title']}")
        
        content.append(f"**URL**: {metadata.get('url', 'No URL')}")
        
        # Empty line before content
        content.append("")
        
        # Include the actual text content that matched
        if metadata.get("chunk_text"):
            chunk_text = metadata["chunk_text"]
            content.append(chunk_text)
        
        panel_content = "\n".join(content)
        console.print(Panel(Markdown(panel_content), title=title, border_style="green", expand=True))
    
    def search(self, query: str) -> Tuple[List[Dict[str, Any]], str]:
        """
        Execute a complete search flow:
        1. Generate embedding for the query
        2. Search Pinecone for similar vectors
        3. Format search results
        4. Generate a ChatGPT response
        
        Returns the raw results and the response.
        """
        logger.info(f"Searching for: {query}")
        
        # Generate embedding for the query
        query_embedding = self.generate_embedding(query)
        
        # Search Pinecone
        results = self.search_pinecone(query_embedding)
        
        # Format search results for ChatGPT
        formatted_results = self.format_search_results(results)
        
        # Generate ChatGPT response
        response = self.generate_chat_response(query, formatted_results)
        
        return results, response

def interactive_mode(searcher: PineconeSearcher):
    """Run the search in interactive mode."""
    console.print(Panel.fit(
        "Virginia Data Explorer\n\nSearch for datasets using natural language queries. Type 'exit' to quit.",
        title="Interactive Search",
        border_style="blue"
    ))
    
    while True:
        # Get user query
        query = console.input("\n[bold blue]Search query[/] (or 'exit' to quit): ")
        
        if query.lower() in ('exit', 'quit', 'q'):
            console.print("[yellow]Goodbye![/]")
            break
        
        if not query.strip():
            continue
        
        try:
            with console.status("[bold green]Searching...[/]"):
                results, response = searcher.search(query)
            
            # Display results table
            searcher.display_results_table(results)
            
            # Display ChatGPT response
            console.print(Panel(Markdown(response), title="AI Response", border_style="cyan", expand=True))
            
            # Offer to show full details
            if results:
                while True:
                    detail_input = console.input("\nView full details for a result? (1-5 or 'n' to skip): ")
                    if detail_input.lower() in ('n', 'no', 'skip'):
                        break
                    
                    try:
                        result_idx = int(detail_input) - 1
                        if 0 <= result_idx < len(results):
                            searcher.display_full_result(results[result_idx])
                        else:
                            console.print("[red]Invalid result number[/]")
                    except ValueError:
                        console.print("[red]Please enter a number or 'n'[/]")
        
        except Exception as e:
            console.print(f"[red]Error:[/] {str(e)}")

def main():
    """Main entry point for the Pinecone search tool."""
    parser = argparse.ArgumentParser(
        description="Search Virginia Open Data Portal datasets using Pinecone vector database and ChatGPT"
    )
    
    parser.add_argument(
        "--index-name", 
        default=DEFAULT_INDEX_NAME,
        help=f"Name of the Pinecone index (default: {DEFAULT_INDEX_NAME})"
    )
    
    parser.add_argument(
        "--namespace", 
        default=DEFAULT_NAMESPACE,
        help=f"Pinecone namespace to use (default: {DEFAULT_NAMESPACE})"
    )
    
    parser.add_argument(
        "--embedding-model", 
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"OpenAI embedding model to use (default: {DEFAULT_EMBEDDING_MODEL})"
    )
    
    parser.add_argument(
        "--chat-model", 
        default=DEFAULT_CHAT_MODEL,
        help=f"OpenAI chat model to use (default: {DEFAULT_CHAT_MODEL})"
    )
    
    parser.add_argument(
        "--top-k", 
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of results to return (default: {DEFAULT_TOP_K})"
    )
    
    parser.add_argument(
        "--query",
        help="Run a single query (non-interactive mode)"
    )
    
    args = parser.parse_args()
    
    # Check for API keys
    if not os.environ.get("OPENAI_API_KEY"):
        console.print("[red]ERROR:[/] OPENAI_API_KEY environment variable not set.")
        console.print("You can set it in a .env file or use: export OPENAI_API_KEY=your_key_here")
        return 1
    
    if not os.environ.get("PINECONE_API_KEY"):
        console.print("[red]ERROR:[/] PINECONE_API_KEY environment variable not set.")
        console.print("You can set it in a .env file or use: export PINECONE_API_KEY=your_key_here")
        return 1
    
    # Create searcher
    try:
        searcher = PineconeSearcher(
            index_name=args.index_name,
            namespace=args.namespace,
            embedding_model=args.embedding_model,
            chat_model=args.chat_model,
            top_k=args.top_k
        )
        
        # Run in single query or interactive mode
        if args.query:
            # Single query mode
            results, response = searcher.search(args.query)
            searcher.display_results_table(results)
            console.print(Panel(Markdown(response), title="AI Response", border_style="cyan", expand=True))
        else:
            # Interactive mode
            interactive_mode(searcher)
        
    except Exception as e:
        console.print(f"[red]Error:[/] {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    main() 