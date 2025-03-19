"""
Virginia Open Data Portal Web Crawler

This module extracts metadata from the Virginia Open Data Portal by crawling
the website pages rather than using the API. This approach may be useful when
API authentication is not available or when additional metadata is needed.
"""
import os
import time
import json
from typing import List, Dict, Any, Optional
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from dotenv import load_dotenv
import pandas as pd
import re
from pathlib import Path

# Load environment variables
load_dotenv()

class VirginiaPortalCrawler:
    """Crawler for the Virginia Open Data Portal."""

    def __init__(self, output_dir: str = "./data/crawled"):
        """Initialize the crawler.
        
        Args:
            output_dir: Directory to save crawled metadata
        """
        self.base_url = os.getenv('VIRGINIA_PORTAL_BASE_URL', 'https://data.virginia.gov')
        self.start_page = int(os.getenv('CRAWLER_START_PAGE', 1))
        self.end_page = int(os.getenv('CRAWLER_END_PAGE', 10))
        self.delay = float(os.getenv('CRAWLER_DELAY', 1.0))
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Session for persistent connection
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def get_dataset_urls(self, page: int) -> List[str]:
        """Get all dataset URLs from a single page.
        
        Args:
            page: Page number to crawl
        
        Returns:
            List of dataset URLs
        """
        url = f"{self.base_url}/dataset?page={page}"
        print(f"Fetching datasets from page {page}: {url}")
        
        response = self.session.get(url)
        if response.status_code != 200:
            print(f"Failed to fetch page {page}: {response.status_code}")
            return []
        
        soup = BeautifulSoup(response.text, 'html.parser')
        dataset_links = soup.select('.dataset-heading a')
        
        return [f"{self.base_url}{a['href']}" for a in dataset_links]
    
    def extract_dataset_metadata(self, url: str) -> Dict[str, Any]:
        """Extract metadata from a dataset page.
        
        Args:
            url: URL of the dataset page
        
        Returns:
            Dictionary containing dataset metadata
        """
        print(f"Extracting metadata from: {url}")
        response = self.session.get(url)
        if response.status_code != 200:
            print(f"Failed to fetch dataset {url}: {response.status_code}")
            return {}
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Basic metadata
        dataset_id = url.split('/')[-1]
        title = soup.select_one('.heading')
        title_text = title.text.strip() if title else "Unknown Title"
        
        # Description
        notes = soup.select_one('.notes')
        description = notes.text.strip() if notes else ""
        
        # Tags
        tags = [tag.text.strip() for tag in soup.select('.tag')]
        
        # Resources
        resources = []
        resource_items = soup.select('.resource-item')
        for item in resource_items:
            resource_url_elem = item.select_one('.resource-url-analytics')
            resource_url = resource_url_elem.get('href') if resource_url_elem else None
            
            resource_format_elem = item.select_one('.format-label')
            resource_format = resource_format_elem.text.strip() if resource_format_elem else "Unknown"
            
            resource_name_elem = item.select_one('.heading')
            resource_name = resource_name_elem.text.strip() if resource_name_elem else "Unknown"
            
            if resource_url:
                resources.append({
                    'name': resource_name,
                    'url': resource_url,
                    'format': resource_format
                })
        
        # Additional metadata
        metadata_items = {}
        additional_info = soup.select('.additional-info th, .additional-info td')
        current_key = None
        
        for i, item in enumerate(additional_info):
            if i % 2 == 0:  # This is a header/key
                current_key = item.text.strip().lower().replace(' ', '_')
            else:  # This is a value
                if current_key:
                    metadata_items[current_key] = item.text.strip()
        
        # Extract organization if available
        org_elem = soup.select_one('.organization-name')
        organization = org_elem.text.strip() if org_elem else None
        
        return {
            'id': dataset_id,
            'title': title_text,
            'description': description,
            'tags': tags,
            'resources': resources,
            'resource_count': len(resources),
            'url': url,
            'organization': organization,
            'metadata': metadata_items
        }
    
    def run(self) -> None:
        """Run the crawler to extract metadata from all datasets."""
        all_datasets = []
        
        # Iterate through all pages
        for page in range(self.start_page, self.end_page + 1):
            dataset_urls = self.get_dataset_urls(page)
            
            if not dataset_urls:
                print(f"No datasets found on page {page}, stopping.")
                break
            
            print(f"Found {len(dataset_urls)} datasets on page {page}")
            
            # Extract metadata from each dataset
            for url in tqdm(dataset_urls):
                try:
                    metadata = self.extract_dataset_metadata(url)
                    if metadata:
                        # Save individual dataset metadata
                        dataset_id = metadata['id']
                        with open(f"{self.output_dir}/{dataset_id}.json", 'w') as f:
                            json.dump(metadata, f, indent=2)
                        
                        all_datasets.append(metadata)
                    
                    # Respect rate limits
                    time.sleep(self.delay)
                except Exception as e:
                    print(f"Error processing {url}: {str(e)}")
            
            # Save progress after each page
            with open(f"{self.output_dir}/all_datasets.json", 'w') as f:
                json.dump(all_datasets, f, indent=2)
            
            print(f"Saved {len(all_datasets)} datasets so far.")
            
            # Respect rate limits between pages
            time.sleep(self.delay * 2)
        
        print(f"Crawler completed. Extracted metadata from {len(all_datasets)} datasets.")
        
        # Create a CSV summary
        df = pd.DataFrame([{
            'id': d['id'],
            'title': d['title'],
            'description': d.get('description', '')[:100] + '...',
            'tag_count': len(d.get('tags', [])),
            'resource_count': d.get('resource_count', 0),
            'organization': d.get('organization', 'Unknown')
        } for d in all_datasets])
        
        df.to_csv(f"{self.output_dir}/dataset_summary.csv", index=False)
        print(f"Summary CSV saved to {self.output_dir}/dataset_summary.csv")

def main():
    """Run the crawler as a standalone script."""
    crawler = VirginiaPortalCrawler()
    crawler.run()

if __name__ == "__main__":
    main() 