# Virginia Data Portal - Knowledge Base Search & Chat

This project provides tools to embed data from the Virginia Data Portal into a vector database (Pinecone) and perform semantic search and chat-based querying with an interactive web UI powered by GPT-4o.

## Features

- **Data Embedding**: Process JSON data from Virginia Data Portal and store embeddings in Pinecone
- **Semantic Search**: Search the embedded data using natural language queries
- **Interactive Chat UI**: Web interface for querying Virginia data with context-aware responses
- **Advanced Formatting**: Displays dataset information with proper formatting and links
- **GPT-4o Integration**: Uses advanced language models for high-quality responses

## Setup

### Prerequisites

- Python 3.8+ for the backend scripts
- Node.js 14+ for the chat UI
- Pinecone account for vector database
- OpenAI API key for embeddings and chat

### Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/V-GPT.git
cd V-GPT
```

2. Install Python dependencies:

```bash
pip install openai pinecone-client python-dotenv tqdm flask cors
```

3. Install Node.js dependencies for the chat UI:

```bash
cd public/chat-ui
npm install
```

4. Create a `.env` file in the root directory with your API keys:

```
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=us-east-1-aws
PINECONE_INDEX_NAME=virginia-data-portal
EMBEDDING_MODEL=text-embedding-3-large
CHAT_MODEL=gpt-4o
```

## Usage

### 1. Embed Data

Process JSON files and store embeddings in Pinecone:

```bash
python embed_json_files.py
```

### 2. Start the Chat UI Server

```bash
cd public/chat-ui
node server.js
```

Then open your browser to `http://localhost:3000` to interact with the Virginia Data Portal Chat UI.

### 3. Command Line Tools

#### Search JSON Data

```bash
python search_json_data.py --query "your search query"
```

#### Chat with Data (Terminal)

```bash
python chat_with_data.py
```

## Scripts Overview

### Backend Scripts

- `embed_json_files.py`: Processes JSON data and creates embeddings in Pinecone
- `search_json_data.py`: Command-line tool for semantic search
- `chat_with_data.py`: Terminal-based interactive chat interface

### Web Interface

- `public/chat-ui/server.js`: Node.js server for the chat interface
- `public/chat-ui/app.js`: Client-side JavaScript for the chat UI
- `public/chat-ui/index.html`: HTML structure for the chat UI
- `public/chat-ui/styles.css`: Styling for the chat interface

## Advanced Options

### Embedding Options

- `--batch-size`: Number of items to process in one batch (default: 100)
- `--chunk-size`: Maximum characters per text chunk (default: 1000)

### Search Options

- `--top-k`: Number of results to return (default: 5)
- `--output`: Save search results to a file

### Chat Model Options

- `--chat-model`: Specify which OpenAI model to use (default: gpt-4o)

## Deployment

For production deployment:

1. Set up a proper web server (Nginx, Apache) to serve the static files
2. Use a process manager like PM2 to keep the Node.js server running
3. Consider containerizing the application with Docker for easier deployment

## Troubleshooting

- If you encounter context length errors, the system will automatically try to reduce context size
- For Pinecone connection issues, check your API key and environment settings
- Run diagnostics from the chat UI by clicking the server check button

## License

[MIT License](LICENSE)

## Acknowledgements

- Virginia Data Portal for providing open data APIs
- OpenAI for GPT models and embeddings
- Pinecone for vector database services 