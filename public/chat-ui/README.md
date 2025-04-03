# Virginia Data Portal Chat Interface

A web-based chat interface that allows users to interact with the Virginia Data Portal using natural language. This application uses OpenAI and Pinecone to provide relevant and accurate responses based on the embedded data.

## Features

- Modern, responsive chat interface
- Integration with OpenAI for natural language understanding
- Integration with Pinecone for vector search of relevant information
- Support for both client-side and server-side API keys
- Fallback to direct OpenAI calls if the server API fails

## Prerequisites

- Node.js (v18 or higher)
- NPM or Yarn
- OpenAI API key
- Pinecone API key and active index with embeddings

## Installation

1. Navigate to the project directory:
   ```
   cd public/chat-ui
   ```

2. Install dependencies:
   ```
   npm install
   ```

3. Start the server:
   ```
   npm start
   ```

4. Open your browser and navigate to:
   ```
   http://localhost:3000
   ```

## Configuration

You can configure the application in two ways:

### 1. Server-side Environment Variables

For production deployments or to provide default keys, set these environment variables:

- `OPENAI_API_KEY`: Your OpenAI API key
- `PINECONE_API_KEY`: Your Pinecone API key
- `PINECONE_ENVIRONMENT`: Your Pinecone environment (default: "us-east-1-aws")
- `PINECONE_INDEX`: Your Pinecone index name (default: "virginia-data-portal")

With server-side configuration, users won't need to enter their own API keys.

### 2. Client-side Configuration

Users can still provide their own API keys through the interface:

1. Click the settings (gear) icon in the top right corner
2. Enter your OpenAI API key
3. Enter your Pinecone API key
4. (Optional) Adjust the Pinecone environment and index name
5. Click "Save Settings"

Client-side settings will be saved in the browser's local storage for convenience.

## Deployment on Render

This application is designed to be easily deployed on [Render](https://render.com/):

1. Create a new Web Service
2. Connect your GitHub repository
3. Set the following:
   - **Build Command**: `cd public/chat-ui && npm install`
   - **Start Command**: `cd public/chat-ui && node server.js`
4. Add the environment variables:
   - `OPENAI_API_KEY`
   - `PINECONE_API_KEY`
   - `PINECONE_ENVIRONMENT`
   - `PINECONE_INDEX`
5. Deploy the service

## Usage

1. Type your question in the text input at the bottom of the screen
2. Press Enter or click the send button
3. The system will search the Virginia Data Portal knowledge base for relevant information
4. A response will be generated based on the retrieved information

## Development

To run the server in development mode with auto-restart:

```
npm run dev
```

## Architecture

- **Frontend**: HTML, CSS, and JavaScript
- **Backend**: Node.js with Express
- **Vector Database**: Pinecone
- **Language Model**: OpenAI (GPT-4 or similar)

The system works by converting user queries into embeddings, searching for relevant information in Pinecone, and then using OpenAI to generate a coherent response based on the retrieved context.

## Environment Variables

The application reads the following environment variables:

| Variable Name | Required | Default | Description |
|---------------|----------|---------|-------------|
| `PORT` | No | 3000 | Port to run the server on |
| `OPENAI_API_KEY` | Yes* | - | OpenAI API key |
| `PINECONE_API_KEY` | Yes* | - | Pinecone API key |
| `PINECONE_ENVIRONMENT` | No | us-east-1-aws | Pinecone environment |
| `PINECONE_INDEX` | No | virginia-data-portal | Pinecone index name |

*Either server-side or client-side keys must be provided.

## Security Note

This application allows API keys to be stored in the browser's local storage. While convenient, this approach has security considerations:

- For production environments, server-side API keys are recommended
- Client-side keys are stored in localStorage, which can be accessed by JavaScript on the same domain
- In a production setting with public access, consider:
  1. Using only server-side keys with proper authentication
  2. Implementing rate limiting
  3. Setting up proper CORS policies 