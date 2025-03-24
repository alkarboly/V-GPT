#!/usr/bin/env python3
"""
Markdown to JSON Processor for Virginia Data Portal

This script:
1. Scans through markdown files created by the crawler
2. Identifies files containing data dictionaries
3. Extracts the structured data from tables
4. Creates tagged and chunked JSON files for search indexing
"""

import os
import sys
import json
import logging
import re
from pathlib import Path
import markdown
from bs4 import BeautifulSoup
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("prepare_json.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("markdown_to_json_processor")

# Set up paths
script_path = Path(__file__).resolve().parent
project_root = script_path.parent
markdown_path = project_root / "data" / "datasets"
json_path = project_root / "data" / "json"
json_path.mkdir(parents=True, exist_ok=True)

def extract_tables_from_markdown(markdown_text):
    """Extract all tables from a markdown file, preserving their context."""
    # Convert markdown to HTML
    html = markdown.markdown(markdown_text, extensions=['tables'])
    soup = BeautifulSoup(html, 'html.parser')
    
    # Find all tables in the HTML
    tables = soup.find_all('table')
    
    # Extract tables with their preceding headers for context
    extracted_tables = []
    
    for table in tables:
        # Try to find a preceding header
        header = None
        prev_elem = table.find_previous(['h1', 'h2', 'h3', 'h4', 'h5'])
        if prev_elem:
            header = prev_elem.text.strip()
        
        # Extract rows from the table
        rows = []
        header_row = []
        
        # Extract header row (th elements)
        for th in table.find_all('th'):
            header_row.append(th.text.strip())
        
        if header_row:
            rows.append(header_row)
        
        # Extract data rows (td elements)
        for tr in table.find_all('tr'):
            # Skip if this is a header row we already processed
            if tr.find('th'):
                continue
                
            row = []
            for td in tr.find_all('td'):
                row.append(td.text.strip())
            
            if row:  # Only add non-empty rows
                rows.append(row)
        
        # Add the extracted table data
        if len(rows) > 1:  # Only include tables with at least a header row and one data row
            extracted_tables.append({
                'context': header,
                'rows': rows
            })
    
    return extracted_tables

def extract_dataset_metadata(markdown_text):
    """Extract basic dataset metadata from the markdown."""
    metadata = {
        'title': None,
        'description': None,
        'organization': None,
        'tags': [],
        'url': None
    }
    
    # Convert markdown to HTML for easier parsing
    html = markdown.markdown(markdown_text)
    soup = BeautifulSoup(html, 'html.parser')
    
    # Extract title (h1)
    h1 = soup.find('h1')
    if h1:
        metadata['title'] = h1.text.strip()
    
    # Try to extract description from first p after h2 that says "Description"
    desc_header = soup.find(lambda tag: tag.name == 'h2' and 'Description' in tag.text)
    if desc_header:
        desc_p = desc_header.find_next('p')
        if desc_p:
            metadata['description'] = desc_p.text.strip()
    
    # Extract other metadata from strong/bold elements
    patterns = {
        'Organization:': 'organization',
        'Tags:': 'tags',
        'Source URL:': 'url'
    }
    
    for strong in soup.find_all('strong'):
        text = strong.text.strip()
        for pattern, field in patterns.items():
            if text.startswith(pattern):
                # Get the parent element
                parent = strong.parent
                if parent:
                    # Get text after the strong element
                    if field == 'tags':
                        # For tags, split by commas
                        tags_text = parent.text.replace(text, '').strip()
                        metadata[field] = [tag.strip() for tag in tags_text.split(',')]
                    else:
                        metadata[field] = parent.text.replace(text, '').strip()
    
    return metadata

def has_data_dictionary(markdown_text):
    """Check if the markdown contains a data dictionary."""
    # Look for tables that appear to be data dictionaries
    html = markdown.markdown(markdown_text, extensions=['tables'])
    soup = BeautifulSoup(html, 'html.parser')
    
    # Common headers that indicate a data dictionary
    patterns = [
        'Column Information',
        'Data Dictionary',
        'Column Name',
        'Field'
    ]
    
    # Check if any headers contain patterns
    for header in soup.find_all(['h4', 'h5']):
        if any(pattern in header.text for pattern in patterns):
            return True
    
    # Check table headers
    for table in soup.find_all('table'):
        header_row = table.find('tr')
        if header_row:
            headers = [th.text.strip().lower() for th in header_row.find_all(['th', 'td'])]
            # Check if headers look like a data dictionary
            if 'column' in ' '.join(headers) and ('type' in ' '.join(headers) or 'description' in ' '.join(headers)):
                return True
    
    return False

def extract_column_info(tables):
    """Extract column information from tables that appear to be data dictionaries."""
    column_info = []
    
    for table in tables:
        context = table.get('context', '')
        rows = table.get('rows', [])
        
        # Skip tables without enough rows
        if len(rows) < 2:
            continue
        
        # Check if this looks like a data dictionary table
        header_row = rows[0]
        header_text = ' '.join(header_row).lower()
        
        # Check if the context or headers suggest this is a column info table
        is_column_table = (
            'column' in header_text or 
            'field' in header_text or
            'Column Information' in context or
            'Data Dictionary' in context
        )
        
        if not is_column_table:
            continue
        
        # Try to identify column indices
        name_idx = None
        type_idx = None
        desc_idx = None
        sample_idx = None
        
        for i, header in enumerate(header_row):
            header_lower = header.lower()
            if 'name' in header_lower or 'column' in header_lower or 'field' in header_lower:
                name_idx = i
            elif 'type' in header_lower:
                type_idx = i
            elif 'description' in header_lower or 'desc' in header_lower or 'notes' in header_lower:
                desc_idx = i
            elif 'sample' in header_lower or 'value' in header_lower or 'example' in header_lower:
                sample_idx = i
        
        # Skip if we can't identify a name column
        if name_idx is None:
            continue
        
        # Process data rows
        for row in rows[1:]:
            if len(row) <= name_idx:
                continue  # Skip rows that don't have enough cells
                
            column = {
                'name': row[name_idx]
            }
            
            if type_idx is not None and type_idx < len(row):
                column['type'] = row[type_idx]
                
            if desc_idx is not None and desc_idx < len(row):
                column['description'] = row[desc_idx]
                
            if sample_idx is not None and sample_idx < len(row):
                # Clean up sample values
                samples_text = row[sample_idx]
                # Remove markdown code ticks
                samples_text = samples_text.replace('`', '')
                column['sample_values'] = samples_text
            
            column_info.append(column)
    
    return column_info

def process_markdown_file(file_path):
    """Process a single markdown file and extract searchable data."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            markdown_text = f.read()
        
        # Extract the dataset ID from the filename
        dataset_id = file_path.stem
        
        # Extract basic metadata
        dataset_metadata = extract_dataset_metadata(markdown_text)
        dataset_metadata['id'] = dataset_id
        
        # Check if it has a data dictionary
        if not has_data_dictionary(markdown_text):
            logger.info(f"No data dictionary found in {file_path.name}, skipping")
            return None
        
        # Extract tables from the markdown
        tables = extract_tables_from_markdown(markdown_text)
        logger.info(f"Found {len(tables)} tables in {file_path.name}")
        
        # Extract column information
        column_info = extract_column_info(tables)
        logger.info(f"Extracted information for {len(column_info)} columns")
        
        if not column_info:
            logger.info(f"No usable column information found in {file_path.name}, skipping")
            return None
        
        # Create simplified dataset record - just metadata and columns
        dataset_record = {
            'id': dataset_id,
            'title': dataset_metadata.get('title'),
            'description': dataset_metadata.get('description', ''),
            'organization': dataset_metadata.get('organization', ''),
            'tags': dataset_metadata.get('tags', []),
            'url': dataset_metadata.get('url', ''),
            'columns': []
        }
        
        # Add columns to the dataset record
        for column in column_info:
            dataset_record['columns'].append({
                'name': column.get('name', ''),
                'type': column.get('type', ''),
                'description': column.get('description', ''),
                'sample_values': column.get('sample_values', '')
            })
        
        return dataset_record
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        return None

def main():
    """Main function to process all markdown files and create JSON output."""
    print("\n=== Virginia Data Portal Markdown to JSON Processor ===")
    
    # Get all markdown files
    markdown_files = list(markdown_path.glob('*.md'))
    logger.info(f"Found {len(markdown_files)} markdown files to process")
    
    # Counters for statistics
    processed_count = 0
    with_dictionary_count = 0
    
    # All datasets for the combined search index
    all_datasets = []
    
    # Process each file
    progress_bar = tqdm(markdown_files, desc="Processing markdown files", unit="file")
    
    for file_path in progress_bar:
        try:
            dataset_record = process_markdown_file(file_path)
            processed_count += 1
            
            if dataset_record:
                # This file has a data dictionary
                with_dictionary_count += 1
                
                # Save individual dataset JSON
                dataset_id = file_path.stem
                json_file_path = json_path / f"{dataset_id}.json"
                
                with open(json_file_path, 'w', encoding='utf-8') as f:
                    json.dump(dataset_record, f, indent=2)
                
                # Add to combined index
                all_datasets.append(dataset_record)
                
                # Update progress bar
                progress_bar.set_postfix(
                    with_dict=f"{with_dictionary_count}/{processed_count}",
                    columns=len(dataset_record.get('columns', []))
                )
                
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
    
    # Save all datasets to a single search index file
    search_index_path = json_path / "search_index.json"
    with open(search_index_path, 'w', encoding='utf-8') as f:
        json.dump(all_datasets, f, indent=2)
    
    # Print summary
    print("\n=== Processing Complete ===")
    print(f"Total markdown files processed: {processed_count}")
    print(f"Files with data dictionaries: {with_dictionary_count}")
    print(f"Individual JSON files saved to: {json_path}")
    print(f"Combined search index saved to: {search_index_path}")

if __name__ == "__main__":
    main() 