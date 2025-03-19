import os
import requests
import pandas as pd
from typing import List, Dict, Any
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from chromadb.config import Settings
from tqdm import tqdm
import json

# Load environment variables
load_dotenv()

class VirginiaDataPortalExtractor:
    def __init__(self):
        self.base_url = os.getenv('VIRGINIA_PORTAL_BASE_URL', 'https://data.virginia.gov')
        self.api_token = os.getenv('VIRGINIA_PORTAL_APP_TOKEN')
        self.headers = {
            'Authorization': self.api_token,
            'Content-Type': 'application/json'
        }
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.chroma_client = chromadb.Client(Settings(
            persist_directory=os.getenv('CHROMA_PERSIST_DIRECTORY', './chroma_db')
        ))
        # Get or create collection
        try:
            self.collection = self.chroma_client.get_collection("virginia_datasets")
            print("Using existing collection: virginia_datasets")
        except ValueError:
            self.collection = self.chroma_client.create_collection(
                name="virginia_datasets",
                metadata={"hnsw:space": "cosine"}
            )
            print("Created new collection: virginia_datasets")

    def fetch_datasets(self) -> List[Dict[str, Any]]:
        """Fetch all datasets from Virginia's Open Data Portal using CKAN API."""
        endpoint = f"{self.base_url}/api/3/action/package_list"
        response = requests.get(endpoint, headers=self.headers)
        response.raise_for_status()
        
        if not response.json().get('success', False):
            error = response.json().get('error', {})
            raise Exception(f"API Error: {error.get('message', 'Unknown error')}")
            
        dataset_names = response.json().get('result', [])
        datasets = []
        
        print(f"Found {len(dataset_names)} datasets, fetching details...")
        for name in tqdm(dataset_names):
            try:
                dataset = self.fetch_dataset_details(name)
                datasets.append(dataset)
            except Exception as e:
                print(f"Error fetching dataset {name}: {str(e)}")
        
        return datasets
        
    def fetch_dataset_details(self, dataset_name: str) -> Dict[str, Any]:
        """Fetch detailed information about a specific dataset."""
        endpoint = f"{self.base_url}/api/3/action/package_show"
        params = {'id': dataset_name}
        
        response = requests.get(endpoint, params=params, headers=self.headers)
        response.raise_for_status()
        
        if not response.json().get('success', False):
            error = response.json().get('error', {})
            raise Exception(f"API Error: {error.get('message', 'Unknown error')}")
            
        return response.json().get('result', {})

    def extract_metadata(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant metadata from a dataset."""
        resources = dataset.get('resources', [])
        resource_urls = [r.get('url') for r in resources if r.get('url')]
        
        tags = [tag.get('name') for tag in dataset.get('tags', []) if tag.get('name')]
        
        return {
            'id': dataset.get('id'),
            'name': dataset.get('name'),
            'title': dataset.get('title'),
            'description': dataset.get('notes', ''),
            'tags': tags,
            'urls': resource_urls,
            'resource_count': len(resources),
            'last_updated': dataset.get('metadata_modified'),
            'created': dataset.get('metadata_created'),
            'organization': dataset.get('organization', {}).get('name'),
            'license_id': dataset.get('license_id'),
            'source_page': f"{self.base_url}/dataset/{dataset.get('name')}"
        }

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using OpenAI's API."""
        response = self.openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding

    def process_dataset(self, dataset: Dict[str, Any]) -> None:
        """Process a single dataset: extract metadata, generate embedding, and store in ChromaDB."""
        metadata = self.extract_metadata(dataset)
        
        # Combine relevant fields for embedding
        text_for_embedding = f"{metadata['title']} {metadata['description']} {' '.join(metadata['tags'])}"
        embedding = self.generate_embedding(text_for_embedding)
        
        # Store in ChromaDB
        self.collection.add(
            embeddings=[embedding],
            documents=[text_for_embedding],
            metadatas=[metadata],
            ids=[metadata['id']]
        )
        return metadata

    def run_pipeline(self):
        """Run the complete pipeline: fetch, process, and store datasets."""
        print("Fetching datasets from Virginia Open Data Portal...")
        datasets = self.fetch_datasets()
        
        print(f"Processing {len(datasets)} datasets...")
        processed = []
        for dataset in tqdm(datasets):
            try:
                metadata = self.process_dataset(dataset)
                processed.append(metadata)
            except Exception as e:
                print(f"Error processing dataset {dataset.get('id', 'unknown')}: {str(e)}")
        
        print(f"Pipeline completed successfully! Processed {len(processed)} datasets.")
        return processed

    def search_similar_datasets(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for similar datasets using semantic similarity."""
        query_embedding = self.generate_embedding(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        return results

if __name__ == "__main__":
    extractor = VirginiaDataPortalExtractor()
    extractor.run_pipeline() 