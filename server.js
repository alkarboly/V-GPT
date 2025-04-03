const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const path = require('path');
const OpenAI = require('openai');
const dotenv = require('dotenv');
const { Pinecone } = require('@pinecone-database/pinecone');

// Load environment variables from .env file
dotenv.config({ path: path.resolve(__dirname, '../../.env') });

const app = express();
const PORT = process.env.PORT || 3000;

// API keys from environment variables
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const PINECONE_API_KEY = process.env.PINECONE_API_KEY;
const PINECONE_ENVIRONMENT = process.env.PINECONE_ENVIRONMENT || 'us-east-1-aws';
const PINECONE_INDEX = process.env.PINECONE_INDEX_NAME || 'virginia-data-portal';
const PINECONE_NAMESPACE = process.env.PINECONE_NAMESPACE || '';

console.log('Environment variables loaded:');
console.log('- OPENAI_API_KEY:', OPENAI_API_KEY ? 'Present (masked)' : 'Not found');
console.log('- PINECONE_API_KEY:', PINECONE_API_KEY ? 'Present (masked)' : 'Not found');
console.log('- PINECONE_ENVIRONMENT:', PINECONE_ENVIRONMENT);
console.log('- PINECONE_INDEX:', PINECONE_INDEX);
console.log('- PINECONE_NAMESPACE:', PINECONE_NAMESPACE);

// Validate required environment variables
if (!OPENAI_API_KEY) {
    console.error('ERROR: OPENAI_API_KEY environment variable is not set');
    process.exit(1);
}

// Initialize OpenAI client
const openai = new OpenAI({
    apiKey: OPENAI_API_KEY
});

// Initialize Pinecone client if API key is available
let pinecone = null;
let usePinecone = false;

if (PINECONE_API_KEY) {
    try {
        // Using the latest Pinecone API format
        pinecone = new Pinecone({
            apiKey: PINECONE_API_KEY
        });
        usePinecone = true;
        console.log('Pinecone client initialized successfully');
    } catch (error) {
        console.error('Error initializing Pinecone client:', error);
        console.log('Continuing without Pinecone integration');
    }
} else {
    console.log('No Pinecone API key provided, continuing without Pinecone integration');
}

// Middleware
app.use(cors());
app.use(bodyParser.json({limit: '10mb'}));  // Increase limit for larger payloads
app.use(express.static(path.join(__dirname)));

// Serve the main HTML file
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

// Add ping endpoint to check server health
app.get('/ping', (req, res) => {
    res.status(200).json({ status: 'ok', message: 'Server is running' });
});

// Add diagnostics endpoint to check connections
app.get('/api/diagnostics', async (req, res) => {
    try {
        const diagnostics = {
            server: 'ok',
            openai: false,
            pinecone: false,
            usePinecone: usePinecone
        };
        
        // Check OpenAI
        try {
            const openaiModels = await openai.models.list();
            diagnostics.openai = true;
            diagnostics.openaiModels = openaiModels.data.slice(0, 5).map(m => m.id);
        } catch (err) {
            diagnostics.openaiError = err.message;
        }
        
        // Check Pinecone if available
        if (usePinecone && pinecone) {
            try {
                // Try to list indexes
                let indexes = [];
                try {
                    indexes = await pinecone.listIndexes();
                    diagnostics.pineconeIndexes = indexes.map(idx => ({
                        name: idx.name
                    }));
                } catch (listErr) {
                    console.log('Error listing indexes:', listErr.message);
                    diagnostics.listIndexesError = listErr.message;
                }
                
                // Try to connect directly to our known index
                try {
                    console.log(`Checking direct connection to: ${PINECONE_INDEX}`);
                    const index = pinecone.Index(PINECONE_INDEX);
                    const stats = await index.describeIndexStats();
                    
                    console.log('Successfully connected to index:', stats);
                    diagnostics.pinecone = true;
                    diagnostics.directConnectionSuccess = true;
                    diagnostics.indexStats = stats;
                    
                    // Add this index to the list if it's not already there
                    if (!diagnostics.pineconeIndexes) {
                        diagnostics.pineconeIndexes = [];
                    }
                    
                    // Check if this index is already in the list
                    const exists = diagnostics.pineconeIndexes.some(idx => 
                        idx.name === PINECONE_INDEX);
                        
                    if (!exists) {
                        diagnostics.pineconeIndexes.push({
                            name: PINECONE_INDEX,
                            direct: true
                        });
                    }
                    
                } catch (directErr) {
                    console.error('Error connecting directly to index:', directErr);
                    diagnostics.directConnectionError = directErr.message;
                }
                
                // Set overall Pinecone status
                diagnostics.pinecone = diagnostics.pinecone || diagnostics.directConnectionSuccess;
                diagnostics.pineconeIndex = PINECONE_INDEX;
                diagnostics.pineconeNamespace = PINECONE_NAMESPACE;
                
            } catch (err) {
                diagnostics.pineconeError = err.message;
            }
        }
        
        res.json(diagnostics);
    } catch (err) {
        console.error('Unhandled error in diagnostics:', err);
        res.status(500).json({ error: err.message });
    }
});

// Helper function to generate embeddings
async function generateEmbedding(text) {
    try {
        const embeddingResponse = await openai.embeddings.create({
            model: "text-embedding-ada-002",
            input: text
        });
        return embeddingResponse.data[0].embedding;
    } catch (err) {
        console.error('Error generating embedding:', err);
        throw err;
    }
}

// Helper function to search Pinecone
async function searchPinecone(embedding, indexName, topK = 5) {
    try {
        // Connect to the index
        console.log(`Connecting to Pinecone index: ${PINECONE_INDEX}`);
        const index = pinecone.Index(PINECONE_INDEX);
        
        // Query the index with the latest API format
        const queryResponse = await index.query({
            vector: embedding,
            topK: topK,
            includeValues: true,
            includeMetadata: true,
            namespace: PINECONE_NAMESPACE
        });
        
        return queryResponse;
    } catch (err) {
        console.error('Error searching Pinecone:', err);
        throw err;
    }
}

// Chat API endpoint
app.post('/api/chat', async (req, res) => {
    try {
        console.log('Received chat request');
        
        const { query, useEmbeddings = true } = req.body;

        // Validate required fields
        if (!query) {
            return res.status(400).json({ error: 'Missing query parameter' });
        }
        
        // Determine if we should use embeddings
        const shouldUseEmbeddings = useEmbeddings && usePinecone;
        
        let context = '';
        
        // If using embeddings, search Pinecone
        if (shouldUseEmbeddings) {
            try {
                console.log('Using embeddings to search Pinecone');
                
                // Generate embedding for query
                const embedding = await generateEmbedding(query);
                console.log('Generated embedding for query');
                
                // Search Pinecone
                const searchResults = await searchPinecone(embedding, PINECONE_INDEX);
                console.log(`Found ${searchResults.matches ? searchResults.matches.length : 0} matches`);
                
                // Extract context from search results
                if (searchResults.matches && searchResults.matches.length > 0) {
                    context = searchResults.matches
                        .map(match => match.metadata.text)
                        .join('\n\n');
                    console.log('Extracted context from search results');
                } else {
                    console.log('No matches found in search results');
                    context = 'No specific information was found in the Virginia Data Portal for this query.';
                }
            } catch (err) {
                console.error('Error using embeddings:', err);
                console.log('Falling back to direct chat');
                // Continue without embeddings
            }
        }

        // Prepare the chat prompt
        const messages = [
            {
                role: 'system',
                content: shouldUseEmbeddings && context 
                    ? `You are a helpful assistant for the Virginia Data Portal. 
                      Use the following information from our knowledge base to answer the user's question. 
                      If the information doesn't contain the answer, say you don't know but try to be helpful.
                      
                      Knowledge base information:
                      ${context}`
                    : `You are a helpful assistant for the Virginia Data Portal. 
                      Answer the user's questions as best you can. If you don't know something specific, 
                      you can say so while being helpful and friendly.`
            },
            {
                role: 'user',
                content: query
            }
        ];

        // Generate response using OpenAI
        console.log('Generating response with OpenAI...');
        let chatResponse;
        try {
            chatResponse = await openai.chat.completions.create({
                model: 'gpt-4',  // or another appropriate model
                messages: messages,
                temperature: 0.7
            });
            console.log('Response generated successfully');
        } catch (err) {
            console.error('Error generating response with OpenAI:', err);
            return res.status(500).json({ 
                error: 'Failed to generate response with OpenAI', 
                details: err.message 
            });
        }

        // Send the response back to the client
        console.log('Sending response to client');
        res.json({
            response: chatResponse.choices[0].message.content,
            usedEmbeddings: shouldUseEmbeddings && context !== ''
        });

    } catch (error) {
        console.error('Unhandled error in chat request:', error);
        res.status(500).json({
            error: 'An error occurred while processing your request',
            details: error.message
        });
    }
});

// Start the server
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
    console.log(`Open http://localhost:${PORT} in your browser`);
    
    // Log configuration status (without exposing actual keys)
    console.log('\nServer configuration:');
    console.log(`- OpenAI API Key: ${OPENAI_API_KEY ? 'Set ✓' : 'Not set ✗'}`);
    console.log(`- Pinecone API Key: ${PINECONE_API_KEY ? 'Set ✓' : 'Not set ✗'}`);
    console.log(`- Pinecone Integration: ${usePinecone ? 'Enabled ✓' : 'Disabled ✗'}`);
    if (usePinecone) {
        console.log(`- Pinecone Environment: ${PINECONE_ENVIRONMENT}`);
        console.log(`- Pinecone Index: ${PINECONE_INDEX}`);
        console.log(`- Pinecone Namespace: ${PINECONE_NAMESPACE}`);
    }
}); 