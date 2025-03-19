# VGPT - Virginia Open Data Portal Metadata Extractor

This project extracts metadata from datasets on Virginia's Open Data Portal, generates embeddings using OpenAI's API, and stores them in a vector database for semantic search capabilities.

## Features

- Fetches metadata from Virginia's Open Data Portal using their CKAN API
- Extracts key metadata including title, description, tags, and URLs
- Generates embeddings using OpenAI's text-embedding-ada-002 model
- Stores embeddings and metadata in ChromaDB for efficient similarity search
- Provides semantic search functionality for finding related datasets
- Includes a FastAPI server for easy integration with N8N.io workflows
- Web crawler for scraping dataset metadata directly from the portal website

## Project Structure

```
VGPT/
├── data/                    # Data storage directory
│   └── crawled/             # Crawled metadata storage
├── docs/                    # Documentation
├── public/                  # Public assets
├── src/                     # Source code
│   ├── api/                 # API endpoints
│   │   └── endpoints.py     # FastAPI endpoints
│   ├── data/                # Data handling utilities
│   ├── utils/               # Utility functions
│   │   └── crawler.py       # Web crawler for metadata extraction
│   ├── main.py              # Main entry point
│   └── metadata_extractor.py # Core metadata extraction logic
├── .env.example             # Example environment variables
├── .gitignore               # Git ignore file
└── requirements.txt         # Project dependencies
```

## Prerequisites

- Python 3.8+
- OpenAI API key
- Virginia Open Data Portal API Token (optional if using web crawler)
- ChromaDB (included in requirements)

## Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and fill in your API keys:
   ```bash
   cp .env.example .env
   ```

## Configuration

Edit the `.env` file with your API keys and settings:
- `OPENAI_API_KEY`: Your OpenAI API key
- `VIRGINIA_PORTAL_APP_TOKEN`: Your Virginia Open Data Portal API Token
- `VIRGINIA_PORTAL_BASE_URL`: The base URL for the Virginia Open Data Portal
- `CHROMA_PERSIST_DIRECTORY`: Directory where ChromaDB will store its data
- `CRAWLER_START_PAGE`: Starting page number for the web crawler
- `CRAWLER_END_PAGE`: Ending page number for the web crawler
- `CRAWLER_DELAY`: Delay between requests in seconds

### Getting a Virginia Open Data Portal API Token

1. Visit the Virginia Open Data Portal: https://data.virginia.gov/
2. Create an account or sign in to your existing account
3. Go to your user profile (usually by clicking on your username in the top right)
4. Navigate to "API Tokens" or "Developer Settings"
5. Create a new API token
6. Copy this token to your `.env` file as `VIRGINIA_PORTAL_APP_TOKEN`

API authentication is done with the Authorization header as described in the [CKAN API documentation](https://docs.ckan.org/en/2.9/api/).

## Usage

### Extract Metadata Using API

Run the metadata extraction pipeline using the CKAN API:
```bash
python -m src.main extract
```

### Crawl Website for Metadata

If you want to extract metadata by crawling the website instead of using the API:
```bash
python -m src.main crawl --start-page 1 --end-page 10 --output ./data/crawled
```

### Generate Embeddings from Crawled Data

After crawling the website, generate embeddings from the crawled metadata:
```bash
python -m src.main embed-crawled --input ./data/crawled
```

### Search Datasets

Search for datasets using semantic search:
```bash
python -m src.main search "education funding data"
```

### Start API Server

Start the FastAPI server for N8N integration:
```bash
python -m src.main api
```

The API will be available at http://localhost:8000 with documentation at http://localhost:8000/docs

## N8N Integration

To integrate with N8N:

1. Create a new workflow in N8N
2. Add an HTTP Request node to call the search endpoint
3. Configure the node with:
   - Method: POST
   - URL: http://localhost:8000/search
   - Body: JSON with query string
   ```json
   {
     "query": "education funding data",
     "limit": 5
   }
   ```
4. Add additional nodes to process and display the results

## License

MIT License 