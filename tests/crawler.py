"""
Enhanced script to crawl Virginia Open Data Portal and extract column information.

This script:
1. Loads datasets from Virginia Open Data Portal
2. Extracts metadata for each dataset
3. Attempts to identify column names and data types when available
4. Saves each dataset as a separate JSON file with the enhanced metadata

Ethical Crawling Practices Implemented:
- Proper user agent identification with contact info
- Rate limiting with randomized delays between requests
- Request caching to reduce server load
- Exponential backoff for failed requests
- Respect for server resources with concurrency limits
"""

import os
import sys
import json
import time
import random
import requests
import pandas as pd
import io
from pathlib import Path
from bs4 import BeautifulSoup
import re
import logging
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
cache_path = project_root / "data" / "cache"
data_path.mkdir(parents=True, exist_ok=True)
cache_path.mkdir(parents=True, exist_ok=True)

print(f"Data will be saved to: {data_path}")
print(f"Cache will be stored in: {cache_path}")

# Base URL for the Virginia Open Data Portal
base_url = "https://data.virginia.gov"
page_url = f"{base_url}/dataset?page=1"

# Crawler configuration
CRAWLER_CONFIG = {
    # Contact information (adjust with your actual info)
    "user_agent": "Datathon Research Crawler",
    
    # Rate limiting
    "min_delay": 2.0,    # Minimum delay between requests in seconds
    "max_delay": 5.0,    # Maximum delay between requests in seconds
    "resource_delay": 3.0,  # Delay when downloading resources like CSVs
    
    # Retry configuration
    "max_retries": 3,    # Maximum number of retries for failed requests
    "retry_backoff": 2,  # Exponential backoff factor
    
    # Request caching
    "use_cache": True,   # Whether to use caching
    "cache_expire": 60 * 60 * 24,  # Cache expiration in seconds (24 hours)
    
    # Crawl limits
    "max_datasets": 50,  # Maximum number of datasets to process
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

def rate_limited_request(session, url, stream=False, is_resource=False):
    """Make a rate-limited request with proper caching and respect for the server."""
    # Create a cache key from the URL
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    path = parsed_url.path
    
    cache_key = f"{domain}{path}".replace('/', '_').replace(':', '_')
    cache_file = cache_path / f"{cache_key}.cache"
    
    # Check if cached response exists and is valid
    if CRAWLER_CONFIG["use_cache"] and cache_file.exists():
        cache_age = time.time() - cache_file.stat().st_mtime
        if cache_age < CRAWLER_CONFIG["cache_expire"]:
            logger.info(f"Using cached response for: {url}")
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                # Return the content directly as a string if we're using cached data
                # This allows code to use response.text directly
                return cached_data["content"]
            except Exception as e:
                logger.error(f"Error reading cache file: {str(e)}")
    
    # Apply rate limiting with randomized delay
    if is_resource:
        delay = CRAWLER_CONFIG["resource_delay"]
    else:
        delay = random.uniform(CRAWLER_CONFIG["min_delay"], CRAWLER_CONFIG["max_delay"])
    
    logger.info(f"Rate limiting: Waiting {delay:.2f}s before request to {domain}")
    time.sleep(delay)
    
    # Make the request
    try:
        response = session.get(url, stream=stream)
        
        # Cache the response if caching is enabled and request succeeded
        if CRAWLER_CONFIG["use_cache"] and response.status_code == 200 and not stream:
            with open(cache_file, 'w', encoding='utf-8') as f:
                cache_data = {
                    "url": url,
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                    "content": response.text,
                    "timestamp": time.time()
                }
                json.dump(cache_data, f, ensure_ascii=False)
        
        return response
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {url} - {str(e)}")
        return None

def get_dataset_urls(base_page_url, start_page, end_page, session=None):
    """Get dataset URLs from a range of pages."""
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
        
        # Response can be either a string (from cache) or a response object
        if isinstance(response, str):
            html_content = response
        else:
            # It's a response object, check status code
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
        
        logger.info(f"  Found {len(page_dataset_urls)} datasets on page {page_num}")
        all_dataset_urls.extend(page_dataset_urls)
    
    logger.info(f"Total datasets collected from pages {start_page}-{end_page}: {len(all_dataset_urls)}")
    return all_dataset_urls, session

def extract_column_info_from_csv(url, session):
    """Extract column information by examining CSV metadata without downloading the entire file."""
    try:
        logger.info(f"  Getting CSV metadata from: {url}")
        
        # Instead of downloading the CSV, just extract metadata from the HTML page
        logger.info(f"  Skipping CSV download as requested, relying on HTML metadata only")
        return None
    except Exception as e:
        logger.error(f"  Error analyzing CSV metadata: {str(e)}")
        return None

def extract_column_info_from_html(soup):
    """Try to extract column information from the HTML page itself."""
    try:
        # Look for data dictionaries or column information in tables
        data_dict_tables = soup.select('.table.table-striped.table-bordered')
        
        # Method 1: Extract from tables with clear column headers
        columns = []
        for table in data_dict_tables:
            # Check if this is a data dictionary table
            headers = [th.text.strip().lower() for th in table.select('th')]
            
            if any(header in ['field', 'column', 'variable', 'name'] for header in headers):
                # This might be a data dictionary table
                rows = table.select('tr')
                
                # Skip header row
                for row in rows[1:]:
                    cells = row.select('td')
                    if len(cells) >= 2:
                        col_name = cells[0].text.strip()
                        # Try to get type/description from the second column
                        col_type = cells[1].text.strip() if len(cells) > 1 else "Unknown"
                        
                        columns.append({
                            "name": col_name,
                            "type": col_type,
                            "description": cells[2].text.strip() if len(cells) > 2 else ""
                        })
        
        # Method 2: Look for metadata in structured div elements
        if not columns:
            # Look for .dataset-details-sidebar which sometimes contains field info
            details_section = soup.select('.dataset-details-sidebar, .additional-info, .notes')
            
            for section in details_section:
                # Look for field definitions - often in definition lists
                dl_elements = section.select('dl')
                for dl in dl_elements:
                    dt_elements = dl.select('dt')
                    dd_elements = dl.select('dd')
                    
                    for i, dt in enumerate(dt_elements):
                        if i < len(dd_elements):
                            term = dt.text.strip()
                            definition = dd_elements[i].text.strip()
                            
                            # If this looks like a field definition
                            if "field" in term.lower() or "column" in term.lower():
                                # Try to parse field type and name
                                field_parts = definition.split(':')
                                if len(field_parts) >= 2:
                                    col_name = field_parts[0].strip()
                                    col_type = field_parts[1].strip()
                                else:
                                    col_name = definition
                                    col_type = "Unknown"
                                
                                columns.append({
                                    "name": col_name,
                                    "type": col_type,
                                    "description": ""
                                })
        
        # Method 3: Look for structured metadata in the page content
        if not columns:
            # Sometimes field info is in paragraphs with specific formatting
            content_sections = soup.select('.notes p, .resource-description p')
            
            for p in content_sections:
                text = p.text.strip()
                
                # Look for patterns like "field1 (type): description, field2 (type): description"
                field_pattern = r'([A-Za-z0-9_]+)\s*\(([^)]+)\)\s*:'
                matches = re.findall(field_pattern, text)
                
                for match in matches:
                    col_name = match[0].strip()
                    col_type = match[1].strip()
                    
                    columns.append({
                        "name": col_name,
                        "type": col_type,
                        "description": ""
                    })
                
                # Also look for table-like structures in text (field: type, field: type)
                if not matches and ":" in text:
                    lines = text.split(",")
                    for line in lines:
                        if ":" in line:
                            parts = line.split(":")
                            if len(parts) >= 2:
                                col_name = parts[0].strip()
                                col_desc = parts[1].strip()
                                
                                # Check if this looks like a field definition
                                if len(col_name) < 50 and not col_name.endswith(("ing", "ed", "tion")):
                                    columns.append({
                                        "name": col_name,
                                        "type": "Unknown",
                                        "description": col_desc
                                    })
        
        # Method 4: Check for any tables that might have column headers 
        if not columns:
            all_tables = soup.select('table')
            for table in all_tables:
                header_row = table.select('tr:first-child th')
                if header_row and len(header_row) > 1:
                    for header in header_row:
                        header_text = header.text.strip()
                        if header_text and len(header_text) < 40:  # Reasonable column name length
                            columns.append({
                                "name": header_text,
                                "type": "Unknown",
                                "description": ""
                            })
        
        if columns:
            # Remove duplicates based on column name
            unique_columns = []
            seen_names = set()
            for col in columns:
                if col["name"].lower() not in seen_names:
                    seen_names.add(col["name"].lower())
                    unique_columns.append(col)
            
            return {
                "column_count": len(unique_columns),
                "columns": unique_columns,
                "extracted_from": "HTML data dictionary"
            }
        
        return None
    except Exception as e:
        logger.error(f"  Error extracting column info from HTML: {str(e)}")
        return None

def extract_column_info_from_preview(session, dataset_id, resource_id):
    """Extract column information from the dataset preview page without downloading the actual file."""
    preview_url = f"{base_url}/dataset/{dataset_id}/resource/{resource_id}/preview"
    logger.info(f"  Checking preview page for column info: {preview_url}")
    
    try:
        # Get the preview page
        response = rate_limited_request(session, preview_url)
        
        if not response:
            logger.error(f"  Failed to fetch preview page: No response")
            return None
        
        # Response can be either a string (from cache) or a response object
        if isinstance(response, str):
            html_content = response
        else:
            # It's a response object, check status code
            if response.status_code != 200:
                logger.error(f"  Preview page returned status: {response.status_code}")
                return None
            html_content = response.text
        
        # Parse the HTML
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # First look for the data dictionary table which is the most reliable source
        logger.info(f"  Searching for data dictionary table...")
        dictionary_heading = soup.find(lambda tag: tag.name in ['h2', 'h3', 'h4'] and 
                                       ('Data Dictionary' in tag.text or 'Data Field' in tag.text))
        dictionary_table = None
        if dictionary_heading:
            logger.info(f"  Found data dictionary heading: {dictionary_heading.text.strip()}")
            # Look for a table after the Data Dictionary heading
            dictionary_table = dictionary_heading.find_next('table')
            if not dictionary_table:
                # Try to find a table within the same section or div
                parent_section = dictionary_heading.parent
                if parent_section:
                    dictionary_table = parent_section.find('table')
        
        # If we found a data dictionary table, extract column info from it
        if dictionary_table:
            logger.info(f"  Found data dictionary table, extracting column information...")
            columns = []
            rows = dictionary_table.select('tr')
            
            # First row contains headers
            headers = [th.text.strip().lower() for th in rows[0].select('th')]
            if not headers:
                headers = [td.text.strip().lower() for td in rows[0].select('td')]
            
            logger.info(f"  Data dictionary headers found: {headers}")
            
            # Skip header row and process data rows
            for row in rows[1:]:
                cells = row.select('td')
                if len(cells) >= 2:
                    col_data = {}
                    for i, cell in enumerate(cells):
                        if i < len(headers):
                            col_data[headers[i]] = cell.text.strip()
                    
                    # Extract the important fields
                    name = col_data.get('column', '')
                    if not name and 'field' in headers:
                        name = col_data.get('field', '')
                    if not name and 'name' in headers:
                        name = col_data.get('name', '')
                    
                    # Extract type and description
                    type_val = col_data.get('type', '')
                    if not type_val and 'data type' in headers:
                        type_val = col_data.get('data type', '')
                    if not type_val:
                        type_val = 'string'
                    
                    description = col_data.get('description', '')
                    
                    # Add to columns list if we have at least a name
                    if name:
                        columns.append({
                            "name": name,
                            "type": type_val,
                            "description": description
                        })
            
            if columns:
                logger.info(f"  Successfully extracted {len(columns)} columns from data dictionary table")
                return {
                    "column_count": len(columns),
                    "columns": columns,
                    "extracted_from": "Data Dictionary table"
                }
            else:
                logger.info(f"  Failed to extract column information from data dictionary table")
        
        # Now, try to find the data preview table
        logger.info(f"  Searching for preview table...")
        
        # First, look for the table container with a specific class
        preview_container = soup.select_one('.resource-view, .data-viewer-info, .ckanext-datapreview, .preview-container')
        
        # If we found a container, try to get the table inside it
        preview_table = None
        if preview_container:
            logger.info(f"  Found preview container with class: {preview_container.get('class')}")
            tables = preview_container.select('table')
            if tables:
                preview_table = tables[0]
                logger.info(f"  Found table inside preview container")
        
        # If we didn't find a table in a container, try directly looking for a preview table 
        if not preview_table:
            logger.info(f"  Looking for table with common preview classes...")
            preview_table = soup.select_one('.table.table-striped.table-bordered, .recline-data-explorer, .preview-table')
            if preview_table:
                logger.info(f"  Found table with preview class")
        
        # If still not found, look for any table with many rows (likely the data table)
        if not preview_table:
            logger.info(f"  Looking for any table with sufficient rows...")
            all_tables = soup.select('table')
            for table in all_tables:
                rows = table.select('tr')
                if len(rows) > 3:  # More than header + 2 data rows
                    preview_table = table
                    logger.info(f"  Found table with {len(rows)} rows")
                    break
        
        if not preview_table:
            logger.info(f"  No preview table found on the preview page, checking for iframes...")
            
            # Try to find any embedded iframe that might contain the preview
            iframes = soup.select('iframe')
            if iframes:
                logger.info(f"  Found {len(iframes)} iframes, checking their content...")
                
                for iframe in iframes:
                    if iframe.get('src'):
                        iframe_src = iframe.get('src')
                        if iframe_src.startswith('/'):
                            iframe_src = f"{base_url}{iframe_src}"
                        
                        logger.info(f"  Checking iframe content at: {iframe_src}")
                        iframe_response = rate_limited_request(session, iframe_src)
                        
                        if iframe_response:
                            # Parse iframe content
                            if isinstance(iframe_response, str):
                                iframe_html = iframe_response
                            else:
                                if iframe_response.status_code == 200:
                                    iframe_html = iframe_response.text
                                else:
                                    logger.info(f"  Iframe returned status: {iframe_response.status_code}")
                                    iframe_html = None
                            
                            if iframe_html:
                                iframe_soup = BeautifulSoup(iframe_html, 'html.parser')
                                iframe_tables = iframe_soup.select('table')
                                if iframe_tables:
                                    preview_table = iframe_tables[0]
                                    logger.info(f"  Found table in iframe")
                                    break
            
            # If still no preview table, check if there are any divs with data-* attributes
            if not preview_table:
                logger.info(f"  Looking for data containers...")
                data_divs = soup.select('[data-recline-state], [data-module="datapreview"], [class*="dataexplorer"], [class*="data-preview"]')
                if data_divs:
                    logger.info(f"  Found potential data container, but no visible table - may be dynamically loaded")
                    return None
            
            if not preview_table:
                logger.info(f"  No suitable preview content found")
                return None
        
        # Try to extract the header row - it could be in thead or just the first tr
        header_row = preview_table.select_one('thead tr')
        if not header_row:
            all_rows = preview_table.select('tr')
            if all_rows:
                header_row = all_rows[0]
        
        if not header_row:
            logger.info(f"  No header row found in preview table")
            return None
        
        # Get header cells, which could be th or td elements
        header_cells = header_row.select('th')
        if not header_cells:
            header_cells = header_row.select('td')
        
        # Get all column names from the header cells
        column_headers = [cell.text.strip() for cell in header_cells if cell.text.strip()]
        if not column_headers:
            logger.info(f"  No column headers found in preview table")
            return None
        
        logger.info(f"  Found {len(column_headers)} column headers: {', '.join(column_headers[:5])}{'...' if len(column_headers) > 5 else ''}")
        
        # Get sample rows to determine data types
        sample_rows = []
        data_rows = preview_table.select('tbody tr')
        if not data_rows:
            # If no tbody, use all rows except the first one
            all_rows = preview_table.select('tr')
            if len(all_rows) > 1:
                data_rows = all_rows[1:min(11, len(all_rows))]  # Get up to 10 sample rows
        
        # Get sample rows
        for row in data_rows:
            cells = row.select('td')
            if cells:
                sample_values = [cell.text.strip() for cell in cells]
                if len(sample_values) == len(column_headers):
                    sample_rows.append(sample_values)
        
        logger.info(f"  Found {len(sample_rows)} sample rows")
        
        # Create column info structure
        columns = []
        for i, header in enumerate(column_headers):
            # Get samples for this column from all rows
            samples = [row[i] for row in sample_rows if i < len(row) and row[i]]
            
            # Try to determine data type from sample values
            data_type = "string"  # Default type
            
            # More intelligent type detection
            if samples:
                # Check numbers - ensure we have at least 2 samples and they're all numeric
                numeric_test = all(re.match(r'^-?\d+(\.\d+)?$', s) for s in samples if s)
                if numeric_test and samples:
                    # Check if any have decimals
                    if any('.' in s for s in samples if s):
                        data_type = "float"
                    else:
                        data_type = "integer"
                
                # Date detection - check common date formats
                elif all(re.match(r'^\d{1,4}[-/]\d{1,2}[-/]\d{1,4}', s) for s in samples if s):
                    data_type = "date"
                
                # Boolean detection
                elif all(s.lower() in ['true', 'false', 'yes', 'no', 't', 'f', 'y', 'n', '1', '0'] for s in samples if s):
                    data_type = "boolean"
                
                # Percentage detection 
                elif all(s.endswith('%') or (re.match(r'^\d+(\.\d+)?%?$', s) and float(s.rstrip('%')) <= 100) for s in samples if s):
                    data_type = "percentage"
                
                # ID fields detection - if name contains "id" and values are all numbers or short codes
                elif ("id" in header.lower() or "_id" in header.lower()) and all(re.match(r'^\w+$', s) for s in samples if s):
                    data_type = "id"
                
                # Categorical detection - if values repeat and there aren't too many unique values
                elif len(set(samples)) < min(5, len(samples)) and len(samples) > 4:
                    data_type = "categorical"
                
                # Text detection for longer text
                elif any(len(s) > 50 for s in samples if s):
                    data_type = "text"
            
            # Special case for common column names
            if header.lower() in ['date', 'week end date', 'report date', 'created', 'modified', 'timestamp']:
                data_type = "date"
            elif header.lower() in ['percent', 'percentage', 'rate', 'ratio']:
                data_type = "percentage"
            elif header.lower() in ['count', 'number', 'quantity']:
                data_type = "integer"
            elif header.lower() in ['name', 'title', 'description']:
                data_type = "text"
            elif header.lower() in ['type', 'category', 'level']:
                data_type = "categorical"
            
            columns.append({
                "name": header,
                "type": data_type,
                "sample_values": samples[:3] if samples else []
            })
        
        logger.info(f"  Successfully extracted {len(columns)} columns from preview table")
        return {
            "column_count": len(columns),
            "columns": columns,
            "extracted_from": "Preview page table",
            "sample_rows": len(sample_rows)
        }
    
    except Exception as e:
        logger.error(f"  Error extracting from preview: {str(e)}")
        import traceback
        logger.error(f"  Traceback: {traceback.format_exc()}")
        return None

def extract_dataset_metadata(url, session):
    """Extract metadata from a dataset page including column information when available."""
    logger.info(f"\nExtracting metadata from: {url}")
    
    # Fetch the dataset page
    response = rate_limited_request(session, url)
    
    if not response:
        logger.error(f"Failed to fetch dataset: No response")
        return None
    
    # Response can be either a string (from cache) or a response object
    if isinstance(response, str):
        html_content = response
    else:
        # It's a response object, check status code
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
    resource_count = 0
    
    # Get all resource elements
    resource_elements = soup.select('.resource-item')
    total_resources = min(len(resource_elements), CRAWLER_CONFIG["max_resources_per_dataset"])
    
    # Create a resource progress bar
    resource_progress = tqdm(
        total=total_resources,
        desc=f"Resources for '{title[:30] + '...' if len(title) > 30 else title}'",
        leave=False,
        unit="resource"
    )
    
    # Try to extract column info from the dataset's main page
    global_column_info = extract_column_info_from_html(soup)
    if global_column_info:
        logger.info(f"Found dataset-level column info: {global_column_info.get('column_count')} columns")
    
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
        resource_name = resource_name_elem.text.strip() if resource_name_elem else f"Resource {resource_count+1}"
        
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
        
        # Try to get column information specific to this resource
        column_info = None
        
        # Step 1: Try to get column info from the preview page
        if resource_id:
            logger.info(f"  Attempting to extract column info from preview page for resource: {resource_name}")
            column_info = extract_column_info_from_preview(session, dataset_id, resource_id)
            if column_info:
                logger.info(f"  Successfully extracted column info from preview page: {column_info.get('column_count')} columns")
            else:
                logger.info(f"  Failed to extract column info from preview page for resource: {resource_name}")
        
        # Step 2: If preview didn't work, check if there's a dedicated resource page with more details
        if not column_info and resource_url and resource_url.startswith(base_url):
            try:
                logger.info(f"  Checking resource page for metadata: {resource_name}")
                resource_response = rate_limited_request(session, resource_url)
                
                if resource_response:
                    # Handle cached responses (string) and regular responses
                    if isinstance(resource_response, str):
                        resource_html = resource_response
                    else:
                        if resource_response.status_code != 200:
                            logger.warning(f"  Resource page returned status: {resource_response.status_code}")
                            resource_html = None
                        else:
                            resource_html = resource_response.text
                    
                    if resource_html:
                        resource_soup = BeautifulSoup(resource_html, 'html.parser')
                        column_info = extract_column_info_from_html(resource_soup)
                        if column_info:
                            logger.info(f"  Found column metadata on resource page: {column_info.get('column_count')} columns")
            except Exception as e:
                logger.error(f"  Error checking resource page: {str(e)}")
        
        # Step 3: If we still couldn't get resource-specific column info, use global column info
        if not column_info:
            column_info = global_column_info
        
        if column_info:
            resource_info["column_info"] = column_info
            resource_progress.set_postfix(columns=column_info.get("column_count", 0))
        else:
            resource_progress.set_postfix(columns="N/A")
        
        resources.append(resource_info)
        resource_count += 1
        resource_progress.update(1)
    
    resource_progress.close()
    
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
    
    # Combine column information from all resources
    all_column_info = {}
    for i, resource in enumerate(resources):
        if "column_info" in resource:
            all_column_info[f"resource_{i+1}_{resource['name']}"] = resource["column_info"]
    
    # Metadata
    metadata = {
        "id": dataset_id,
        "title": title,
        "description": description,
        "tags": tags,
        "resources": resources,
        "column_information": all_column_info if all_column_info else None,
        "organization": organization,
        "additional_metadata": metadata_table,
        "url": url,
        "crawl_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Print basic info
    logger.info(f"Extracted: {title}")
    logger.info(f"  Tags: {', '.join(tags[:5])}{'...' if len(tags) > 5 else ''}")
    logger.info(f"  Resources: {len(resources)}")
    if all_column_info:
        total_columns = sum(info.get("column_count", 0) for info in all_column_info.values())
        logger.info(f"  Columns identified: {total_columns}")
    
    return metadata

def save_metadata_as_markdown(metadata, data_path):
    """Save metadata as a Markdown file for better vectorization."""
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
                extraction_source = resource['column_info'].get('extracted_from', 'Unknown')
                sample_rows = resource['column_info'].get('sample_rows', 0)
                md_content += f"#### Columns ({extraction_source}"
                if sample_rows:
                    md_content += f", {sample_rows} sample rows"
                md_content += ")\n\n"
                
                cols = resource['column_info'].get('columns', [])
                if cols:
                    md_content += "| Column Name | Data Type | Sample Values |\n"
                    md_content += "|-------------|-----------|---------------|\n"
                    for col in cols:
                        col_name = col.get('name', 'Unknown')
                        col_type = col.get('type', 'Unknown')
                        samples = col.get('sample_values', [])
                        
                        # Format sample values based on their data type
                        if col_type == "date":
                            sample_str = ", ".join(f"`{s}`" for s in samples[:3]) if samples else ""
                        elif col_type in ["float", "integer", "percentage"]:
                            sample_str = ", ".join(f"`{s}`" for s in samples[:3]) if samples else ""
                        elif col_type == "boolean":
                            sample_str = ", ".join(f"`{s}`" for s in samples[:3]) if samples else ""
                        elif col_type == "categorical":
                            # For categorical, show unique values
                            unique_samples = list(set(samples))
                            sample_str = ", ".join(f"`{s}`" for s in unique_samples[:3]) if unique_samples else ""
                        else:
                            # For text and other types
                            sample_str = ", ".join(s[:50] + "..." if len(s) > 50 else s for s in samples[:2]) if samples else ""
                        
                        md_content += f"| {col_name} | {col_type} | {sample_str} |\n"
                    md_content += "\n"
    
    # Add complete dataset structure section
    if metadata.get("column_information"):
        md_content += f"## Dataset Structure\n\n"
        
        # Get a comprehensive list of all columns from all resources
        all_columns = []
        resource_columns = {}
        
        for resource_name, info in metadata["column_information"].items():
            extraction_source = info.get('extracted_from', 'Unknown')
            cols = info.get('columns', [])
            
            if cols:
                resource_columns[resource_name] = cols
                for col in cols:
                    all_columns.append({
                        "resource": resource_name,
                        "name": col.get('name', 'Unknown'),
                        "type": col.get('type', 'Unknown'),
                        "description": col.get('description', ''),
                        "samples": col.get('sample_values', [])
                    })
        
        # Add a table with all columns
        md_content += "### Column Overview\n\n"
        md_content += "| Column Name | Data Type | Description | Sample Values |\n"
        md_content += "|-------------|-----------|-------------|---------------|\n"
        
        # Sort columns by name for easier reference
        all_columns.sort(key=lambda x: x["name"])
        
        for col in all_columns:
            col_name = col["name"]
            col_type = col["type"]
            description = col["description"] if col["description"] else "No description available"
            
            # Format sample values
            samples = col["samples"]
            if col_type == "date":
                sample_str = ", ".join(f"`{s}`" for s in samples[:2]) if samples else ""
            elif col_type in ["float", "integer", "percentage"]:
                sample_str = ", ".join(f"`{s}`" for s in samples[:2]) if samples else ""
            elif col_type == "boolean":
                sample_str = ", ".join(f"`{s}`" for s in samples[:2]) if samples else ""
            else:
                # For text and other types
                sample_str = ", ".join(s[:30] + "..." if len(s) > 30 else s for s in samples[:2]) if samples else ""
            
            md_content += f"| {col_name} | {col_type} | {description} | {sample_str} |\n"
        
        md_content += "\n\n"
        
        # For each resource, add more detailed information
        for resource_name, cols in resource_columns.items():
            clean_name = resource_name.replace("resource_", "")
            md_content += f"### {clean_name}\n\n"
            
            if cols:
                md_content += "| Column Name | Data Type | Description | Sample Values |\n"
                md_content += "|-------------|-----------|-------------|---------------|\n"
                
                for col in cols:
                    col_name = col.get('name', 'Unknown')
                    col_type = col.get('type', 'Unknown')
                    description = col.get('description', 'No description available')
                    samples = col.get('sample_values', [])
                    
                    # Format sample values based on type
                    if col_type == "date":
                        sample_str = ", ".join(f"`{s}`" for s in samples[:3]) if samples else ""
                    elif col_type in ["float", "integer", "percentage"]:
                        sample_str = ", ".join(f"`{s}`" for s in samples[:3]) if samples else ""
                    elif col_type == "boolean":
                        sample_str = ", ".join(f"`{s}`" for s in samples[:3]) if samples else ""
                    elif col_type == "categorical":
                        unique_samples = list(set(samples))
                        sample_str = ", ".join(f"`{s}`" for s in unique_samples[:3]) if unique_samples else ""
                    else:
                        sample_str = ", ".join(s[:30] + "..." if len(s) > 30 else s for s in samples[:2]) if samples else ""
                    
                    md_content += f"| {col_name} | {col_type} | {description} | {sample_str} |\n"
                
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

def extract_resource_metadata(resource_url, session):
    """Extract metadata directly from a resource page URL."""
    logger.info(f"\nExtracting metadata from resource: {resource_url}")
    
    # Fetch the resource page
    response = rate_limited_request(session, resource_url)
    
    if not response:
        logger.error(f"Failed to fetch resource: No response")
        return None
    
    # Response can be either a string (from cache) or a response object
    if isinstance(response, str):
        html_content = response
    else:
        # It's a response object, check status code
        if response.status_code != 200:
            logger.error(f"Failed to fetch resource: {response.status_code}")
            return None
        html_content = response.text
    
    # Parse the HTML
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Extract dataset ID and resource ID from URL
    url_parts = resource_url.split('/')
    dataset_id = None
    resource_id = None
    
    for i, part in enumerate(url_parts):
        if part == 'dataset' and i+1 < len(url_parts):
            dataset_id = url_parts[i+1]
        elif part == 'resource' and i+1 < len(url_parts):
            resource_id = url_parts[i+1]
    
    if not dataset_id or not resource_id:
        logger.error(f"Failed to extract dataset_id or resource_id from URL: {resource_url}")
        return None
    
    # Basic metadata - try multiple selectors to find the title
    title_elem = soup.select_one('h1.heading, .resource-item-title, h2.page-heading')
    if not title_elem:
        # If we couldn't find a title element, try to extract from breadcrumbs
        breadcrumbs = soup.select('.breadcrumb li')
        if breadcrumbs and len(breadcrumbs) >= 2:
            title_elem = breadcrumbs[-1]
        # If still nothing, try finding any heading
        elif not title_elem:
            title_elem = soup.select_one('h1, h2')
    
    # If we have a dataset ID, use it to create a title if nothing else worked
    title = title_elem.text.strip() if title_elem else f"Dataset {dataset_id}"
    
    # Clean up the title - sometimes it has extra info in parentheses
    if '(' in title:
        title = title.split('(')[0].strip()
    
    # Description - try standard elements, and also the main content area
    desc_elem = soup.select_one('.notes, .resource-description, .dataset-description, .prose')
    description = desc_elem.text.strip() if desc_elem else ""
    
    # If no description found, try to get it from meta tags
    if not description:
        meta_desc = soup.select_one('meta[name="description"]')
        if meta_desc:
            description = meta_desc.get('content', '')
    
    # Format - look for format indicator
    format_elem = soup.select_one('.format-label, .label-format')
    resource_format = format_elem.text.strip() if format_elem else "Unknown"
    
    # Try to extract column information from the data preview table first
    preview_info = None
    if resource_id:
        preview_info = extract_column_info_from_preview(session, dataset_id, resource_id)
    
    # If that didn't work, try to extract from the current page
    if not preview_info:
        # Check if there's a table on this page
        page_info = extract_column_info_from_html(soup)
        if page_info:
            preview_info = page_info
    
    # Resource metadata
    resource_info = {
        "id": resource_id,
        "name": title,
        "description": description,
        "format": resource_format,
        "url": resource_url,
        "column_info": preview_info
    }
    
    # Metadata for the markdown file
    metadata = {
        "id": f"{dataset_id}_{resource_id}",
        "title": title,
        "description": description,
        "tags": [],
        "resources": [resource_info],
        "column_information": {f"resource_{title}": preview_info} if preview_info else None,
        "organization": None,
        "additional_metadata": {},
        "url": resource_url,
        "crawl_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Print basic info
    logger.info(f"Extracted resource: {title}")
    if preview_info:
        logger.info(f"  Columns identified: {preview_info.get('column_count')}")
        for col in preview_info.get('columns', []):
            logger.info(f"    - {col.get('name')}: {col.get('type')}")
    
    return metadata

def main():
    """Crawl the Virginia Open Data Portal and save datasets as markdown files for vectorization."""
    print("\n=== Virginia Open Data Portal Enhanced Crawler ===")
    print("Crawling pages in batches and creating markdown versions of datasets for vectorization\n")
    
    # Check if a specific resource URL was provided as a command line argument
    if len(sys.argv) > 1:
        # Check if the argument is a URL or a page range
        if sys.argv[1].startswith('http'):
            target_url = sys.argv[1]
            # Determine if this is a resource URL or a dataset URL
            is_resource = 'resource' in target_url
            is_dataset = 'dataset' in target_url and not is_resource
            
            if is_resource:
                print(f"Processing specific resource URL: {target_url}")
                
                # Create a session and process this resource
                session = get_session()
                
                # Extract resource metadata
                metadata = extract_resource_metadata(target_url, session)
                
                if metadata:
                    md_file_path = save_metadata_as_markdown(metadata, data_path)
                    print(f"Successfully saved markdown file: {md_file_path}")
                    print(f"Column information extracted: {bool(metadata.get('column_information'))}")
                    return
                else:
                    print(f"Failed to extract metadata from resource: {target_url}")
                    return
            
            elif is_dataset:
                print(f"Processing specific dataset URL: {target_url}")
                
                # Create a session and process this dataset
                session = get_session()
                
                # Extract dataset metadata
                metadata = extract_dataset_metadata(target_url, session)
                
                if metadata:
                    md_file_path = save_metadata_as_markdown(metadata, data_path)
                    print(f"Successfully saved markdown file: {md_file_path}")
                    
                    # Check if any column information was found
                    column_info_found = False
                    for resource in metadata.get('resources', []):
                        if resource.get('column_info'):
                            column_info_found = True
                            break
                    
                    print(f"Column information extracted: {column_info_found}")
                    return
                else:
                    print(f"Failed to extract metadata from dataset: {target_url}")
                    return
        else:
            # Check if argument is a page range (format: start-end)
            try:
                page_range = sys.argv[1].split('-')
                if len(page_range) == 2:
                    start_page = int(page_range[0])
                    end_page = int(page_range[1])
                else:
                    start_page = int(sys.argv[1])
                    end_page = start_page + 9  # Default to 10 pages per batch
                
                # Set crawler to use these page ranges
                print(f"Processing pages {start_page} to {end_page}")
            except ValueError:
                print(f"Invalid page range format. Use start-end or just start page number.")
                return
    else:
        # Default batch: pages 1-10
        start_page = 1
        end_page = 10
    
    print(f"Using ethical crawling practices:")
    print(f"  - User Agent: {CRAWLER_CONFIG['user_agent']}")
    print(f"  - Rate Limiting: {CRAWLER_CONFIG['min_delay']}-{CRAWLER_CONFIG['max_delay']}s between requests")
    print(f"  - Respecting server resources with backoff and caching")
    print(f"  - No downloading of actual dataset files (CSV, XLS, etc.)")
    print(f"  - Extracting column metadata from dataset preview tables")
    print(f"  - No JSON saving, only markdown files for vectorization")
    print(f"  - Processing pages {start_page} to {end_page} in this batch")
    print(f"  - Limited to {CRAWLER_CONFIG['max_datasets']} datasets per batch, {CRAWLER_CONFIG['max_resources_per_dataset']} resources per dataset\n")
    
    # Create a session for all requests
    session = get_session()
    
    # Get dataset URLs from the specified range of pages
    dataset_urls, session = get_dataset_urls(f"{base_url}/dataset", start_page, end_page, session)
    
    if not dataset_urls:
        logger.error("No datasets found. Exiting.")
        return
    
    # Save dataset URLs to a simple text file instead of JSON, append mode for batch processing
    batch_file_path = data_path / f"dataset_urls_batch_{start_page}-{end_page}.txt"
    with open(batch_file_path, 'w', encoding='utf-8') as f:
        f.write(f"# Virginia Open Data Portal Datasets\n")
        f.write(f"# Crawled on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Pages: {start_page} to {end_page}\n")
        f.write(f"# Total datasets in batch: {len(dataset_urls)}\n\n")
        for url in dataset_urls:
            f.write(f"{url}\n")
    
    # Append to master list
    master_file_path = data_path / "all_dataset_urls.txt"
    append_mode = 'a' if master_file_path.exists() else 'w'
    with open(master_file_path, append_mode, encoding='utf-8') as f:
        if append_mode == 'w':
            f.write(f"# Virginia Open Data Portal Datasets - Master List\n")
            f.write(f"# Started crawling on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        else:
            f.write(f"\n# Batch {start_page}-{end_page} added on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        for url in dataset_urls:
            f.write(f"{url}\n")
    
    # Apply dataset limit if configured
    if CRAWLER_CONFIG["max_datasets"] > 0 and len(dataset_urls) > CRAWLER_CONFIG["max_datasets"]:
        logger.info(f"Limiting to {CRAWLER_CONFIG['max_datasets']} datasets per configuration")
        dataset_urls = dataset_urls[:CRAWLER_CONFIG["max_datasets"]]
    
    # Process each dataset
    saved_md_files = []
    total_datasets = len(dataset_urls)
    
    # Use tqdm for progress tracking
    progress_bar = tqdm(dataset_urls, desc=f"Processing datasets (batch {start_page}-{end_page})", unit="dataset")
    
    # Track preview tables found
    preview_tables_found = 0
    
    for i, url in enumerate(progress_bar):
        try:
            # Update progress bar description with current dataset number
            progress_bar.set_description(f"Dataset {i+1}/{total_datasets} (batch {start_page}-{end_page})")
            
            metadata = extract_dataset_metadata(url, session)
            
            if metadata:
                # Count preview tables
                for resource in metadata.get("resources", []):
                    if resource.get("column_info") and resource["column_info"].get("extracted_from") == "Preview page table":
                        preview_tables_found += 1
                
                # Save only markdown for vectorization
                md_file_path = save_metadata_as_markdown(metadata, data_path)
                saved_md_files.append(md_file_path)
                
                # Update progress bar postfix with success count and preview table count
                progress_bar.set_postfix(saved=f"{len(saved_md_files)}/{i+1}", previews=preview_tables_found)
            
        except Exception as e:
            logger.error(f"Error processing {url}: {str(e)}")
            progress_bar.set_postfix(saved=f"{len(saved_md_files)}/{i+1}", error="Yes")
    
    # Print batch summary
    logger.info(f"\n=== Batch {start_page}-{end_page} Complete ===")
    logger.info(f"Processed {len(dataset_urls)} datasets")
    logger.info(f"Successfully saved {len(saved_md_files)} markdown files for vectorization")
    logger.info(f"Preview tables extracted: {preview_tables_found}")
    logger.info(f"All data saved to: {data_path}")
    
    # Print the command to run the next batch
    next_batch_start = end_page + 1
    next_batch_end = end_page + 10
    print(f"\nTo process the next batch, run:")
    print(f"python tests/crawler.py {next_batch_start}-{next_batch_end}")
    
    # Show datasets with column information from log analysis
    column_datasets = 0
    preview_mention_count = 0
    try:
        for log_line in open("crawler.log", "r").readlines():
            if "Columns identified:" in log_line:
                column_datasets += 1
            if "Successfully extracted" in log_line and "columns from preview table" in log_line:
                preview_mention_count += 1
        
        logger.info(f"Datasets with column information: {column_datasets}")
        logger.info(f"Preview extractions mentioned in logs: {preview_mention_count}")
    except Exception as e:
        logger.error(f"Error analyzing log file: {str(e)}")

# Add a batch runner function to automatically process multiple batches with pauses
def batch_runner(start_batch=1, end_batch=10, batch_size=10, pause_seconds=10):
    """Run multiple batches with pauses between them."""
    print(f"\n=== Batch Runner ===")
    print(f"Processing from page {start_batch} to {start_batch + (end_batch-1)*batch_size}")
    print(f"Using {batch_size} pages per batch with {pause_seconds} second pauses between batches")
    
    for batch in range(start_batch, end_batch + 1):
        batch_start = (batch - 1) * batch_size + 1
        batch_end = batch * batch_size
        
        print(f"\n>>> Starting batch {batch}: pages {batch_start}-{batch_end}")
        
        # Create the command argument
        batch_arg = f"{batch_start}-{batch_end}"
        
        # Import sys and modify argv
        old_argv = sys.argv.copy()
        sys.argv = [sys.argv[0], batch_arg]
        
        try:
            # Run the main function
            main()
        except Exception as e:
            print(f"Error in batch {batch}: {str(e)}")
        finally:
            # Restore argv
            sys.argv = old_argv
        
        # Only pause if this isn't the last batch
        if batch < end_batch:
            print(f"\nPausing for {pause_seconds} seconds before next batch...")
            time.sleep(pause_seconds)

if __name__ == "__main__":
    # Check if the special batch runner mode is activated
    if len(sys.argv) > 1 and sys.argv[1] == "batch-runner":
        # Parse batch runner parameters
        start_batch = 1
        end_batch = 10
        batch_size = 10
        pause_seconds = 10
        
        if len(sys.argv) > 2:
            parts = sys.argv[2].split('-')
            start_batch = int(parts[0])
            end_batch = int(parts[1]) if len(parts) > 1 else start_batch
        
        if len(sys.argv) > 3:
            batch_size = int(sys.argv[3])
        
        if len(sys.argv) > 4:
            pause_seconds = int(sys.argv[4])
        
        # Run in batch mode
        batch_runner(start_batch, end_batch, batch_size, pause_seconds)
    else:
        # Run in regular mode
        main() 