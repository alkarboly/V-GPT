# Virginia Data Portal Crawler

This crawler is designed to ethically extract dataset metadata from the Virginia Open Data Portal. It focuses on accurate extraction of column information and data previews directly from the API.

## Features

- Extracts dataset metadata from Virginia Open Data Portal
- Uses pagination to process multiple pages of datasets
- Extracts detailed column information from the API
- Saves metadata as well-formatted Markdown files
- Implements ethical crawling practices
- Prepares structured JSON from markdown files
- Generates embeddings using OpenAI API
- Stores embeddings in Pinecone vector database
- Provides a web-based search interface

## Ethical Practices

This crawler follows ethical web crawling principles:

- Identifies itself with a proper user agent
- Uses rate limiting to avoid overwhelming the server
- Implements exponential backoff for failed requests
- Only accesses publicly available data
- Restricts the number of requests per session

## Workflow

1. **Data Collection:** Use the crawler to extract metadata from the Virginia Data Portal
2. **Data Processing:** Process markdown files into structured JSON  
3. **Vector Embeddings:** Generate embeddings and store them in Pinecone
4. **Search Interface:** Use the web app to search the dataset collection

## Usage

### Step 1: Crawl Data

```
python crawler.py [page_range]
```

Where:
- `page_range` is optional and can be:
  - A single number (e.g., `3`) to process just that page
  - A range (e.g., `1-5`) to process pages 1 through 5

Example: 
```
python crawler.py 1-3
```

### Step 2: Process Markdown to JSON

After gathering dataset information with the crawler, process the markdown files into structured JSON:

```
python prepare_json.py
```

This script:
- Scans through all markdown files in the `data/datasets` directory
- Identifies files containing data dictionaries (column information)
- Extracts structured data from tables
- Creates JSON files with metadata and column information

The output is stored in `data/json/` with both individual dataset files and a combined `search_index.json`.

### Step 3: Generate Embeddings and Store in Pinecone

Generate vector embeddings and store them in Pinecone for semantic search:

```
python embed_and_store.py --api-key YOUR_OPENAI_KEY --pinecone-key YOUR_PINECONE_KEY
```

You can also set the API keys as environment variables:
```
export OPENAI_API_KEY=your_openai_key
export PINECONE_API_KEY=your_pinecone_key
python embed_and_store.py
```

Additional options:
- `--pinecone-env`: Pinecone environment (default: us-west1-gcp)
- `--index-name`: Pinecone index name (default: virginia-data)
- `--batch-size`: Batch size for processing (default: 50)

### Step 4: Run the Search Application

Start the web-based search interface:

```
export OPENAI_API_KEY=your_openai_key
export PINECONE_API_KEY=your_pinecone_key
export PINECONE_ENV=your_pinecone_env
export PINECONE_INDEX=your_index_name
python app.py
```

Then open http://localhost:5000 in your browser to search the datasets.

## Output

For each dataset, the crawler creates a detailed Markdown file containing:

- Dataset title and description
- Organization information
- Tags
- Available resources
- Detailed column information when available via API
- Sample data when available
- Additional metadata 

Files are saved in the `data/datasets/` directory with the dataset ID as the filename.

## Requirements

- Python 3.7+
- BeautifulSoup4
- Requests
- tqdm (for progress bars)
- Markdown (for JSON processing)
- OpenAI (for embeddings generation)
- Pinecone (for vector database)
- Flask (for web interface)

## Configuration

The crawler's behavior can be configured by modifying the `CRAWLER_CONFIG` dictionary at the top of the script:

- `user_agent`: Identifies the crawler
- `min_delay` and `max_delay`: Rate limiting between requests (in seconds)
- `max_retries` and `retry_backoff`: Controls retry behavior
- `max_datasets`: Maximum number of datasets to process
- `max_resources_per_dataset`: Maximum resources to process per dataset 