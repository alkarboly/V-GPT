"""
Virginia Open Data Portal Crawler - Production Version

This script:
1. Loads datasets from Virginia Open Data Portal with pagination
2. Extracts metadata for each dataset
3. Extracts column information from preview tables when available
4. Saves dataset metadata as Markdown files

Implements ethical crawling practices:
- Proper user agent identification
- Rate limiting with randomized delays between requests
- Exponential backoff for failed requests
"""

import os
import sys
import time
import random
import requests
import logging
from pathlib import Path
from bs4 import BeautifulSoup
import re
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from urllib.parse import urlparse
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("crawler.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("virginia_open_data_crawler")

# Set up paths and make sure the data directory exists
script_path = Path(__file__).resolve().parent
project_root = script_path.parent
data_path = project_root / "data" / "datasets"
data_path.mkdir(parents=True, exist_ok=True)

# Base URL for the Virginia Open Data Portal
base_url = "https://data.virginia.gov"

# Crawler configuration
CRAWLER_CONFIG = {
    # Contact information
    "user_agent": "Virginia Open Data Research Crawler/ Datathon Ahmed.Alkarboly@dmv.virginia.gov",
    
    # Rate limiting
    "min_delay": 0.1,    # Minimum delay between requests in seconds
    "max_delay": 0.3,    # Maximum delay between requests in seconds
    
    # Retry configuration
    "max_retries": 3,    # Maximum number of retries for failed requests
    "retry_backoff": 2,  # Exponential backoff factor
    
    # Crawl limits
    "max_datasets": 15000,  # Maximum number of datasets to process
    "max_resources_per_dataset": 5,  # Maximum number of resources to process per dataset
}

def get_session():
    """Create a requests session with proper configuration for ethical crawling."""
    session = requests.Session()
    
    # Set user agent with contact information
    session.headers.update({
        'User-Agent': CRAWLER_CONFIG["user_agent"]
    })
    
    # Configure automatic retries with exponential backoff
    retry_strategy = Retry(
        total=CRAWLER_CONFIG["max_retries"],
        backoff_factor=CRAWLER_CONFIG["retry_backoff"],
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"]
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session

def rate_limited_request(session, url):
    """Make a rate-limited request with proper backoff for server protection."""
    # Apply rate limiting with randomized delay
    delay = random.uniform(CRAWLER_CONFIG["min_delay"], CRAWLER_CONFIG["max_delay"])
    logger.info(f"Rate limiting: Waiting {delay:.2f}s before request")
    time.sleep(delay)
    
    # Make the request
    try:
        response = session.get(url)
        return response
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {url} - {str(e)}")
        return None

def get_dataset_urls(base_page_url, start_page, end_page, session=None):
    """Get dataset URLs from a range of pages with pagination."""
    logger.info(f"Fetching datasets from pages {start_page} to {end_page}")
    
    # Create a session with proper user agent if not provided
    if session is None:
        session = get_session()
    
    all_dataset_urls = []
    
    # Process each page in the range
    for page_num in range(start_page, end_page + 1):
        page_url = f"{base_page_url}?page={page_num}"
        logger.info(f"Fetching page {page_num}: {page_url}")
        
        # Fetch the page with rate limiting
        response = rate_limited_request(session, page_url)
        
        if not response:
            logger.error(f"Failed to fetch page {page_num}: No response")
            continue
        
        if response.status_code != 200:
            logger.error(f"Failed to fetch page {page_num}: {response.status_code}")
            continue
            
        html_content = response.text
        
        # Parse the HTML
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract dataset links
        dataset_links = soup.select('.dataset-heading a')
        
        # If we found no datasets, this might be the last page
        if not dataset_links:
            logger.info(f"No datasets found on page {page_num}, might be the last page")
            break
        
        # Process all found datasets on this page
        page_dataset_urls = []
        for link in dataset_links:
            full_url = f"{base_url}{link['href']}"
            page_dataset_urls.append(full_url)
        
        logger.info(f"Found {len(page_dataset_urls)} datasets on page {page_num}")
        all_dataset_urls.extend(page_dataset_urls)
    
    logger.info(f"Total datasets collected from pages {start_page}-{end_page}: {len(all_dataset_urls)}")
    return all_dataset_urls, session

def extract_column_info_from_api(session, resource_id):
    """Extract column information directly from the datastore API."""
    api_url = f"{base_url}/api/3/action/datastore_search?resource_id={resource_id}&limit=5"
    logger.info(f"Fetching data from API: {api_url}")
    
    try:
        # Get the API response
        response = rate_limited_request(session, api_url)
        
        if not response or response.status_code != 200:
            logger.error(f"Failed to fetch API data: {response.status_code if response else 'No response'}")
            return None
        
        # Parse the JSON response
        try:
            data = response.json()
            if not data.get('success'):
                logger.error(f"API returned error: {data.get('error', {}).get('message', 'Unknown error')}")
                return None
            
            # Extract the fields and records
            result = data.get('result', {})
            fields = result.get('fields', [])
            records = result.get('records', [])
            total_records = result.get('total', 0)
            
            if not fields or not records:
                logger.error(f"API returned no fields or records")
                return None
            
            logger.info(f"API returned {len(fields)} fields and {len(records)} sample records (total: {total_records})")
            
            # Extract column information
            columns = []
            for field in fields:
                field_id = field.get('id')
                field_type = field.get('type', 'string')
                
                # Get additional info if available
                field_info = field.get('info', {})
                field_label = field_info.get('label', '')
                field_notes = field_info.get('notes', '')
                
                # Get sample values
                sample_values = []
                for record in records:
                    if field_id in record:
                        value = record[field_id]
                        if value:  # Exclude empty values
                            sample_values.append(str(value))
                
                columns.append({
                    "name": field_id,
                    "type": field_type,
                    "description": field_notes,
                    "label": field_label,
                    "sample_values": sample_values[:3]  # Limit to 3 samples
                })
            
            # Convert records to raw data format for display
            raw_data = []
            for record in records:
                row = [str(record.get(field.get('id'), '')) for field in fields]
                raw_data.append(row)
            
            return {
                "column_count": len(columns),
                "columns": columns,
                "extracted_from": "API data",
                "format": "API Data",
                "sample_rows": len(records),
                "total_rows": total_records,
                "raw_data": raw_data,
                "headers": [field.get('id') for field in fields]
            }
            
        except ValueError as e:
            logger.error(f"Error parsing API response: {str(e)}")
            return None
        
    except Exception as e:
        logger.error(f"Error fetching from API: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

def extract_dataset_metadata(url, session):
    """Extract metadata from a dataset page including column information from previews."""
    logger.info(f"\nExtracting metadata from: {url}")
    
    # Fetch the dataset page
    response = rate_limited_request(session, url)
    
    if not response:
        logger.error(f"Failed to fetch dataset: No response")
        return None
    
    if response.status_code != 200:
        logger.error(f"Failed to fetch dataset: {response.status_code}")
        return None
        
    html_content = response.text
    
    # Parse the HTML
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Basic metadata
    dataset_id = url.split('/')[-1]
    title_elem = soup.select_one('h1.heading')
    title = title_elem.text.strip() if title_elem else "Unknown Title"
    
    # Description
    notes_elem = soup.select_one('.notes')
    description = notes_elem.text.strip() if notes_elem else ""
    
    # Tags
    tags = [tag.text.strip() for tag in soup.select('.tag')]
    
    # Resources
    resources = []
    
    # Get all resource elements
    resource_elements = soup.select('.resource-item')
    total_resources = min(len(resource_elements), CRAWLER_CONFIG["max_resources_per_dataset"])
    
    # Process each resource
    for resource_elem in resource_elements[:CRAWLER_CONFIG["max_resources_per_dataset"]]:
        # Get basic resource info
        resource_link = resource_elem.select_one('a')
        resource_url = resource_link.get('href') if resource_link else None
        
        # Make sure the URL is absolute
        if resource_url and not resource_url.startswith('http'):
            resource_url = base_url + resource_url if not resource_url.startswith('/') else base_url + resource_url
        
        # Extract resource ID from URL 
        resource_id = None
        if resource_url:
            url_parts = resource_url.split('/')
            for i, part in enumerate(url_parts):
                if part == 'resource' and i+1 < len(url_parts):
                    resource_id = url_parts[i+1].split('#')[0]  # Remove any fragment
                    break
        
        # Get resource name
        resource_name_elem = resource_elem.select_one('.heading')
        resource_name = resource_name_elem.text.strip() if resource_name_elem else f"Resource {len(resources)+1}"
        
        # Get resource description
        resource_desc_elem = resource_elem.select_one('.description')
        resource_description = resource_desc_elem.text.strip() if resource_desc_elem else ""
        
        # Get resource format
        format_elem = resource_elem.select_one('.format-label')
        resource_format = format_elem.text.strip() if format_elem else "Unknown"
        
        # Basic resource info
        resource_info = {
            "id": resource_id,
            "name": resource_name,
            "description": resource_description,
            "format": resource_format,
            "url": resource_url
        }
        
        # Use only API approach to get column information
        column_info = None
        if resource_id:
            logger.info(f"Extracting column info from API for resource: {resource_name}")
            column_info = extract_column_info_from_api(session, resource_id)
            if column_info:
                logger.info(f"Successfully extracted column info from API: {column_info.get('column_count')} columns")
                resource_info["column_info"] = column_info
        
        resources.append(resource_info)
    
    # Organization
    org_elem = soup.select_one('.organization-name')
    organization = org_elem.text.strip() if org_elem else None
    
    # Additional metadata from the page
    metadata_table = {}
    for item in soup.select('.additional-info th, .additional-info td'):
        if item.name == 'th':
            current_key = item.text.strip().lower().replace(' ', '_')
        elif item.name == 'td' and 'current_key' in locals():
            metadata_table[current_key] = item.text.strip()
    
    # Metadata
    metadata = {
        "id": dataset_id,
        "title": title,
        "description": description,
        "tags": tags,
        "resources": resources,
        "organization": organization,
        "additional_metadata": metadata_table,
        "url": url,
        "crawl_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Print basic info
    logger.info(f"Extracted: {title}")
    logger.info(f"  Tags: {', '.join(tags[:5])}{'...' if len(tags) > 5 else ''}")
    logger.info(f"  Resources: {len(resources)}")
    
    return metadata

def save_metadata_as_markdown(metadata, data_path):
    """Save metadata as a Markdown file."""
    dataset_id = metadata["id"]
    file_path = data_path / f"{dataset_id}.md"
    
    # Generate markdown content
    md_content = f"# {metadata['title']}\n\n"
    
    if metadata.get("description"):
        md_content += f"## Description\n\n{metadata['description']}\n\n"
    
    if metadata.get("organization"):
        md_content += f"**Organization:** {metadata['organization']}\n\n"
    
    if metadata.get("tags") and len(metadata['tags']) > 0:
        md_content += f"**Tags:** {', '.join(metadata['tags'])}\n\n"
    
    if metadata.get("url"):
        md_content += f"**Source URL:** {metadata['url']}\n\n"
    
    # Add resources with emphasis on column information from preview tables
    if metadata.get("resources") and len(metadata['resources']) > 0:
        md_content += f"## Available Resources\n\n"
        for i, resource in enumerate(metadata['resources']):
            md_content += f"### Resource {i+1}: {resource.get('name', 'Unnamed Resource')}\n\n"
            md_content += f"**Format:** {resource.get('format', 'Unknown')}\n\n"
            if resource.get('url'):
                md_content += f"**URL:** {resource['url']}\n\n"
            
            # Add column information if available
            if resource.get('column_info'):
                column_info = resource['column_info']
                extraction_source = column_info.get('extracted_from', 'Unknown')
                data_format = column_info.get('format', 'Unknown')
                
                md_content += f"#### Columns ({extraction_source})\n\n"
                
                # Different handling based on data format
                if 'Excel' in data_format:
                    # Excel data
                    embed_url = column_info.get('embed_url', '')
                    original_url = column_info.get('original_url', '')
                    columns_count = column_info.get('column_count', 0)
                    rows_count = column_info.get('row_count', 0)
                    
                    md_content += f"This resource is an Excel file with {columns_count} columns and {rows_count} rows.\n\n"
                    
                    # Display sample data if available
                    headers = column_info.get('headers', [])
                    sample_data = column_info.get('sample_data', [])
                    
                    if headers and sample_data:
                        md_content += "##### Sample Data:\n\n"
                        
                        # Create markdown table with headers
                        md_content += "| " + " | ".join(headers) + " |\n"
                        md_content += "|" + "|".join(["---" for _ in headers]) + "|\n"
                        
                        # Add sample data rows
                        for row in sample_data:
                            md_content += "| " + " | ".join(row) + " |\n"
                        
                        md_content += "\n"
                    
                    if embed_url:
                        md_content += f"**Embedded Preview URL:** {embed_url}\n\n"
                    
                    if original_url:
                        md_content += f"**Original Excel URL:** {original_url}\n\n"
                
                elif 'API' in data_format:
                    # API data (most detailed)
                    total_rows = column_info.get('total_rows', 0)
                    sample_rows = column_info.get('sample_rows', 0)
                    raw_data = column_info.get('raw_data', [])
                    headers = column_info.get('headers', [])
                    
                    md_content += f"This resource contains {total_rows} total rows. Here are {sample_rows} sample rows:\n\n"
                    
                    # Display column information
                    md_content += "##### Column Information:\n\n"
                    md_content += "| Column Name | Data Type | Description | Sample Values |\n"
                    md_content += "|-------------|-----------|-------------|---------------|\n"
                    
                    for col in column_info.get('columns', []):
                        col_name = col.get('name', 'Unknown')
                        col_type = col.get('type', 'Unknown')
                        description = col.get('description', '')
                        label = col.get('label', '')
                        
                        # Use label as description if available and description is empty
                        if not description and label:
                            description = label
                            
                        samples = col.get('sample_values', [])
                        sample_str = ", ".join(f"`{s}`" for s in samples) if samples else ""
                        
                        md_content += f"| {col_name} | {col_type} | {description} | {sample_str} |\n"
                    
                    md_content += "\n"
                    
                    # Display sample data as a table if available
                    if headers and raw_data:
                        md_content += "##### Sample Data:\n\n"
                        
                        # Create markdown table with headers
                        md_content += "| " + " | ".join(headers) + " |\n"
                        md_content += "|" + "|".join(["---" for _ in headers]) + "|\n"
                        
                        # Add sample data rows (up to 10)
                        for row in raw_data[:10]:
                            # Sanitize row data for markdown table
                            sanitized_row = [cell.replace('|', '\\|') for cell in row]
                            md_content += "| " + " | ".join(sanitized_row) + " |\n"
                        
                        md_content += "\n"
                
                elif 'CSV' in data_format:
                    # CSV data
                    csv_url = column_info.get('csv_download_url', '')
                    sample_rows = column_info.get('sample_rows', 0)
                    raw_data = column_info.get('raw_data', [])
                    headers = column_info.get('headers', [])
                    
                    # Column info table
                    md_content += "##### Column Information:\n\n"
                    md_content += "| Column Name | Data Type | Sample Values |\n"
                    md_content += "|-------------|-----------|---------------|\n"
                    
                    for col in column_info.get('columns', []):
                        col_name = col.get('name', 'Unknown')
                        col_type = col.get('type', 'Unknown')
                        samples = col.get('sample_values', [])
                        
                        # Format sample values based on type
                        if col_type == "date":
                            sample_str = ", ".join(f"`{s}`" for s in samples) if samples else ""
                        elif col_type in ["float", "integer", "percentage"]:
                            sample_str = ", ".join(f"`{s}`" for s in samples) if samples else ""
                        else:
                            # For text and other types
                            sample_str = ", ".join(s[:30] + "..." if len(s) > 30 else s for s in samples) if samples else ""
                        
                        md_content += f"| {col_name} | {col_type} | {sample_str} |\n"
                    
                    md_content += "\n"
                    
                    # Display sample data as a table if available
                    if headers and raw_data:
                        md_content += f"##### Sample Data ({sample_rows} rows shown):\n\n"
                        
                        # Create markdown table with headers
                        md_content += "| " + " | ".join(headers) + " |\n"
                        md_content += "|" + "|".join(["---" for _ in headers]) + "|\n"
                        
                        # Add sample data rows
                        for row in raw_data:
                            # Sanitize row data for markdown table
                            sanitized_row = [cell.replace('|', '\\|') for cell in row]
                            md_content += "| " + " | ".join(sanitized_row) + " |\n"
                        
                        md_content += "\n"
                    
                    if csv_url:
                        md_content += f"**CSV Download URL:** {csv_url}\n\n"
                
                else:
                    # Generic handling for other formats (Data Dictionary, etc.)
                    cols = column_info.get('columns', [])
                    if cols:
                        # Generate detailed column table - different formats based on information available
                        has_description = any(col.get('description') for col in cols)
                        
                        if has_description:
                            # Include description column when available
                            md_content += "| Column Name | Data Type | Description | Sample Values |\n"
                            md_content += "|-------------|-----------|-------------|---------------|\n"
                            for col in cols:
                                col_name = col.get('name', 'Unknown')
                                col_type = col.get('type', 'Unknown')
                                description = col.get('description', '')
                                samples = col.get('sample_values', [])
                                
                                # Format sample values
                                sample_str = ", ".join(s[:30] + "..." if len(s) > 30 else s for s in samples[:2]) if samples else ""
                                
                                md_content += f"| {col_name} | {col_type} | {description} | {sample_str} |\n"
                        else:
                            # Simpler table when no descriptions
                            md_content += "| Column Name | Data Type | Sample Values |\n"
                            md_content += "|-------------|-----------|---------------|\n"
                            for col in cols:
                                col_name = col.get('name', 'Unknown')
                                col_type = col.get('type', 'Unknown')
                                samples = col.get('sample_values', [])
                                
                                # Format sample values based on type
                                if col_type == "date":
                                    sample_str = ", ".join(f"`{s}`" for s in samples[:2]) if samples else ""
                                elif col_type in ["float", "integer", "percentage"]:
                                    sample_str = ", ".join(f"`{s}`" for s in samples[:2]) if samples else ""
                                elif col_type == "boolean":
                                    sample_str = ", ".join(f"`{s}`" for s in samples[:2]) if samples else ""
                                else:
                                    # For text and other types
                                    sample_str = ", ".join(s[:30] + "..." if len(s) > 30 else s for s in samples[:2]) if samples else ""
                                
                                md_content += f"| {col_name} | {col_type} | {sample_str} |\n"
                        
                        md_content += "\n"
    
    # Add additional metadata
    if metadata.get("additional_metadata"):
        md_content += f"## Additional Metadata\n\n"
        for key, value in metadata["additional_metadata"].items():
            md_content += f"**{key.replace('_', ' ').title()}:** {value}\n\n"
    
    # Save the file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    logger.info(f"Saved markdown to: {file_path}")
    return file_path

def main():
    """Crawl the Virginia Open Data Portal and save datasets as Markdown files."""
    print("\n=== Virginia Open Data Portal Crawler - Production Version ===")
    
    # Process command line arguments for page range
    if len(sys.argv) > 1:
        # Check if argument is a page range (format: start-end)
        try:
            page_range = sys.argv[1].split('-')
            if len(page_range) == 2:
                start_page = int(page_range[0])
                end_page = int(page_range[1])
            else:
                start_page = int(sys.argv[1])
                end_page = start_page  # Just process one page
            
            # Set crawler to use these page ranges
            print(f"Processing pages {start_page} to {end_page}")
        except ValueError:
            print(f"Invalid page range format. Use start-end or just start page number.")
            return
    else:
        # Default: page 1
        start_page = 1
        end_page = 1
    
    print(f"Using ethical crawling practices:")
    print(f"  - User Agent: {CRAWLER_CONFIG['user_agent']}")
    print(f"  - Rate Limiting: {CRAWLER_CONFIG['min_delay']}-{CRAWLER_CONFIG['max_delay']}s between requests")
    print(f"  - Respecting server resources with backoff")
    print(f"  - Extracting column metadata from dataset API")
    print(f"  - Saving data as Markdown files for better readability")
    
    # Create a session for all requests
    session = get_session()
    
    # Get dataset URLs from the specified range of pages
    dataset_urls, session = get_dataset_urls(f"{base_url}/dataset", start_page, end_page, session)
    
    if not dataset_urls:
        logger.error("No datasets found. Exiting.")
        return
    
    # Apply dataset limit if configured
    if CRAWLER_CONFIG["max_datasets"] > 0 and len(dataset_urls) > CRAWLER_CONFIG["max_datasets"]:
        logger.info(f"Limiting to {CRAWLER_CONFIG['max_datasets']} datasets per configuration")
        dataset_urls = dataset_urls[:CRAWLER_CONFIG["max_datasets"]]
    
    # Process each dataset
    saved_files = []
    total_datasets = len(dataset_urls)
    
    # Use tqdm for progress tracking
    progress_bar = tqdm(dataset_urls, desc=f"Processing datasets", unit="dataset")
    
    for i, url in enumerate(progress_bar):
        try:
            # Update progress bar description with current dataset number
            progress_bar.set_description(f"Dataset {i+1}/{total_datasets}")
            
            metadata = extract_dataset_metadata(url, session)
            
            if metadata:
                # Save metadata as Markdown
                md_file_path = save_metadata_as_markdown(metadata, data_path)
                saved_files.append(md_file_path)
                
                # Update progress bar postfix with success count
                progress_bar.set_postfix(saved=f"{len(saved_files)}/{i+1}")
            
        except Exception as e:
            logger.error(f"Error processing {url}: {str(e)}")
            progress_bar.set_postfix(saved=f"{len(saved_files)}/{i+1}", error="Yes")
    
    # Print summary
    logger.info(f"\n=== Processing Complete ===")
    logger.info(f"Processed {len(dataset_urls)} datasets")
    logger.info(f"Successfully saved {len(saved_files)} Markdown files")
    logger.info(f"All data saved to: {data_path}")

if __name__ == "__main__":
    main() 