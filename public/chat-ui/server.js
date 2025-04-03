const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const path = require('path');
const OpenAI = require('openai');
const dotenv = require('dotenv');
// Import the Pinecone class as shown in the documentation example
const { Pinecone } = require('@pinecone-database/pinecone');

console.log('=======================================================================');
console.log('Starting Virginia Data Portal Chat Server with Pinecone Vector Search');
console.log('Enhanced with chat_with_data.py functionality');
console.log('=======================================================================');

// Load environment variables from .env file
dotenv.config({ path: path.resolve(__dirname, '../../.env') });

const app = express();
const PORT = process.env.PORT || 3000;

// API keys from environment variables
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const PINECONE_API_KEY = process.env.PINECONE_API_KEY;
const PINECONE_INDEX = process.env.PINECONE_INDEX_NAME || 'virginia-data-portal';
const PINECONE_HOST = "virginia-data-portal-nprks3k.svc.aped-4627-b74a.pinecone.io";

// Update embedding model to match chat_with_data.py
const EMBEDDING_MODEL = process.env.EMBEDDING_MODEL || "text-embedding-3-large";
const CHAT_MODEL = process.env.CHAT_MODEL || "gpt-4o";

console.log('Environment variables loaded:');
console.log('- OPENAI_API_KEY:', OPENAI_API_KEY ? 'Present (masked)' : 'Not found');
console.log('- PINECONE_API_KEY:', PINECONE_API_KEY ? 'Present (masked)' : 'Not found');
console.log('- PINECONE_INDEX:', PINECONE_INDEX);
console.log('- PINECONE_HOST:', PINECONE_HOST);
console.log('- EMBEDDING_MODEL:', EMBEDDING_MODEL);
console.log('- CHAT_MODEL:', CHAT_MODEL);

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
let pineconeIndex = null;
let usePinecone = false;

// Initialize Pinecone with minimal code
async function initPinecone() {
    console.log('🔄 INIT: Starting Pinecone initialization');
    console.log(`🔄 INIT: Environment: ${process.env.PINECONE_ENVIRONMENT || 'Default environment'}`);
    console.log(`🔄 INIT: Target index: ${PINECONE_INDEX}`);
    
    try {
        console.log('🔑 INIT: Checking Pinecone API key:', PINECONE_API_KEY ? 'Key exists' : 'No key');
        
        if (!PINECONE_API_KEY) {
            console.error('❌ INIT ERROR: Pinecone API key is missing');
            return false;
        }
        
        // Create client
        console.log('🔄 INIT: Creating Pinecone client');
        const pc = new Pinecone({ apiKey: PINECONE_API_KEY });
        console.log('✅ INIT: Pinecone client created successfully');
        
        // List indexes
        console.log('📋 INIT: Listing indexes...');
        const listStartTime = Date.now();
        const indexList = await pc.listIndexes();
        const listTime = Date.now() - listStartTime;
        console.log(`✅ INIT: Received index list in ${listTime}ms`);
        
        // Check if our target index exists in the list
        let targetIndexExists = false;
        let indexFound = null;
        
        if (indexList && indexList.indexes) {
            console.log(`📋 INIT: Found ${indexList.indexes.length} indexes`);
            indexFound = indexList.indexes.find(idx => idx.name === PINECONE_INDEX);
            targetIndexExists = !!indexFound;
        } else if (Array.isArray(indexList)) {
            console.log(`📋 INIT: Found ${indexList.length} indexes`);
            indexFound = indexList.find(idx => (idx.name || idx) === PINECONE_INDEX);
            targetIndexExists = !!indexFound;
        }
        
        console.log(`${targetIndexExists ? '✅' : '⚠️'} INIT: Target index ${PINECONE_INDEX} ${targetIndexExists ? 'found' : 'not found'} in index list`);
        
        if (!targetIndexExists) {
            console.error(`❌ INIT ERROR: Index "${PINECONE_INDEX}" not found in your Pinecone account`);
            console.log('📋 INIT: Available indexes:', JSON.stringify(indexList, null, 2));
            return false;
        }
        
        // Connect to the index using its name (this is the preferred and working approach)
        console.log(`🔌 INIT: Connecting to index by name: ${PINECONE_INDEX}`);
        
        try {
            console.log('🔄 INIT: Creating index connection');
            const connectStartTime = Date.now();
            const index = pc.index(PINECONE_INDEX);
            const connectTime = Date.now() - connectStartTime;
            console.log(`✅ INIT: Index connection created in ${connectTime}ms`);
            
            // Test with a simple stats request
            console.log('📊 INIT: Getting index stats...');
            const statsStartTime = Date.now();
            const stats = await index.describeIndexStats();
            const statsTime = Date.now() - statsStartTime;
            console.log(`✅ INIT: Got index stats in ${statsTime}ms`);
            console.log('✅ INIT: Index stats:', JSON.stringify(stats, null, 2));
            
            // Test search functionality with a random vector
            console.log('🔍 INIT: Testing search functionality with random vector...');
            const testVector = Array(1536).fill(0).map(() => Math.random() - 0.5);
            const queryResponse = await index.query({
                vector: testVector,
                topK: 1
            });
            console.log(`✅ INIT: Search test successful, found ${queryResponse.matches ? queryResponse.matches.length : 0} matches`);
            
            // Store the client and index
            pinecone = pc;
            pineconeIndex = index;
            usePinecone = true;
            console.log('✅ INIT: Pinecone integration ENABLED');
            
            return true;
        } catch (error) {
            console.error('❌ INIT ERROR: Failed to connect to Pinecone index:', error);
            console.log('⚠️ INIT: Pinecone integration DISABLED');
            return false;
        }
    } catch (error) {
        console.error('❌ INIT ERROR: Pinecone initialization failed:', error);
        console.log('⚠️ INIT: Pinecone integration DISABLED');
        return false;
    }
}

// Start initialization process
initPinecone().catch(error => {
    console.error('Error in Pinecone initialization:', error);
    console.log('Continuing without Pinecone integration');
});

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
                // First try to list indexes
                try {
                    // List indexes using new format
                    console.log('DIAG: Listing Pinecone indexes...');
                    const indexList = await pinecone.listIndexes();
                    console.log('DIAG: Found Pinecone indexes:', JSON.stringify(indexList, null, 2));
                    
                    // Format indexes for display
                    if (Array.isArray(indexList)) {
                        diagnostics.pineconeIndexes = indexList.map(idx => ({ name: idx.name || idx }));
                    } else if (indexList && indexList.indexes) {
                        diagnostics.pineconeIndexes = indexList.indexes.map(idx => ({ name: idx.name || idx }));
                    } else {
                        diagnostics.pineconeIndexes = [{ name: PINECONE_INDEX }];
                    }
                } catch (listErr) {
                    console.log('DIAG: Error listing indexes:', listErr.message);
                    diagnostics.listIndexesError = listErr.message;
                }
                
                // Try to connect to our index - use the index name approach which works
                try {
                    console.log(`DIAG: Connecting to Pinecone index by name: ${PINECONE_INDEX}`);
                    
                    // Create index reference using the INDEX NAME (not host)
                    const index = pinecone.index(PINECONE_INDEX);
                    
                    // Test with stats
                    console.log('DIAG: Getting index stats...');
                    const stats = await index.describeIndexStats();
                    
                    console.log('DIAG: Successfully connected to Pinecone index:', JSON.stringify(stats, null, 2));
                    diagnostics.pinecone = true;
                    diagnostics.indexStats = stats;
                    
                    // Try a simple query to verify search functionality
                    try {
                        console.log('DIAG: Testing query functionality with random vector...');
                        // Create a random vector with the correct dimension
                        const testVector = Array(1536).fill(0).map(() => Math.random() - 0.5);
                        
                        const queryResponse = await index.query({
                            vector: testVector,
                            topK: 1,
                            includeMetadata: true
                        });
                        
                        console.log('DIAG: Query test successful:', 
                            queryResponse.matches ? `Found ${queryResponse.matches.length} matches` : 'No matches found');
                        
                        // Add query test results to diagnostics
                        diagnostics.queryTest = {
                            success: true,
                            matches: queryResponse.matches ? queryResponse.matches.length : 0
                        };
                    } catch (queryErr) {
                        console.error('DIAG: Query test failed:', queryErr);
                        diagnostics.queryTest = {
                            success: false,
                            error: queryErr.message
                        };
                    }
                } catch (indexErr) {
                    console.error('DIAG: Error connecting to index:', indexErr);
                    diagnostics.indexError = indexErr.message;
                }
                
                // Set overall Pinecone status
                diagnostics.pineconeIndex = PINECONE_INDEX;
                
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

// Helper function to generate embeddings - updated to use the same model as chat_with_data.py
async function generateEmbedding(text) {
    try {
        console.log(`Generating embedding using model: ${EMBEDDING_MODEL}`);
        const embeddingResponse = await openai.embeddings.create({
            model: EMBEDDING_MODEL,
            input: text
        });
        return embeddingResponse.data[0].embedding;
    } catch (err) {
        console.error('Error generating embedding:', err);
        throw err;
    }
}

// Format context for chat - similar to format_context_for_chatgpt in chat_with_data.py
function formatContextForChat(results) {
    console.log('🔄 FORMAT: Starting context formatting');
    
    if (!results || !results.matches || results.matches.length === 0) {
        console.log('ℹ️ FORMAT: No matches to format');
        return "No relevant data found.";
    }
    
    console.log(`🔄 FORMAT: Formatting ${results.matches.length} matches`);
    let context = "Here is some relevant data from the Virginia Data Portal:\n\n";
    
    let extractedTextCount = 0;
    
    // Sort matches by relevance score (highest first)
    const sortedMatches = [...results.matches].sort((a, b) => b.score - a.score);
    
    // Limit the number of matches to process to avoid token limit issues
    const maxMatches = 5; // Process at most 5 matches
    const matchesToProcess = sortedMatches.slice(0, maxMatches);
    
    console.log(`🔄 FORMAT: Processing top ${matchesToProcess.length} matches out of ${results.matches.length} total`);
    
    matchesToProcess.forEach((match, i) => {
        console.log(`🔄 FORMAT: Processing match ${i+1}`);
        
        // Log the full metadata for debugging
        console.log(`🔄 FORMAT: Full metadata for match ${i+1}:`, JSON.stringify(match.metadata, null, 2));
        
        // Extract dataset title/name from metadata using various potential field names
        let datasetName = 'Unknown Dataset';
        if (match.metadata) {
            if (match.metadata.dataset_title) datasetName = match.metadata.dataset_title;
            else if (match.metadata.title) datasetName = match.metadata.title;
            else if (match.metadata.name) datasetName = match.metadata.name;
            else if (match.metadata.dataset_name) datasetName = match.metadata.dataset_name;
            
            console.log(`🔄 FORMAT: Dataset name extracted: "${datasetName}"`);
        }
        
        // Extract organization/source info if available
        let organization = '';
        if (match.metadata && match.metadata.organization) {
            organization = match.metadata.organization;
            console.log(`🔄 FORMAT: Organization extracted: "${organization}"`);
        }
        
        // Extract tags if available
        let tags = [];
        if (match.metadata && match.metadata.tags) {
            if (typeof match.metadata.tags === 'string') {
                try {
                    tags = JSON.parse(match.metadata.tags);
                } catch (e) {
                    tags = match.metadata.tags.split(',').map(tag => tag.trim());
                }
            } else if (Array.isArray(match.metadata.tags)) {
                tags = match.metadata.tags;
            }
            console.log(`🔄 FORMAT: Tags extracted: ${tags.join(', ')}`);
        }
        
        // Extract main content text with better prioritization
        let text = '';
        if (match.metadata) {
            // Try to find specific fields that might contain dataset descriptions
            if (typeof match.metadata.description === 'string' && match.metadata.description.length > 100) {
                text = match.metadata.description;
                console.log('🔄 FORMAT: Using metadata.description');
            } else if (typeof match.metadata.text === 'string') {
                text = match.metadata.text;
                console.log('🔄 FORMAT: Using metadata.text');
            } else if (typeof match.metadata.content === 'string') {
                text = match.metadata.content;
                console.log('🔄 FORMAT: Using metadata.content');
            } else if (typeof match.metadata.chunk_text === 'string') {
                text = match.metadata.chunk_text;
                console.log('🔄 FORMAT: Using metadata.chunk_text');
            } else {
                // Try to find any text field in metadata
                console.log('🔄 FORMAT: Searching for text field in metadata');
                for (const [key, value] of Object.entries(match.metadata)) {
                    if (typeof value === 'string' && value.length > 100) {
                        text = value;
                        console.log(`🔄 FORMAT: Found text in metadata.${key}`);
                        break;
                    }
                }
                
                // If no suitable field found, use the full metadata
                if (!text) {
                    text = JSON.stringify(match.metadata, null, 2);
                    console.log('🔄 FORMAT: Using full metadata JSON');
                }
            }
        } else {
            console.log('⚠️ FORMAT: No metadata available for this match');
        }
        
        // Limit the length of text to avoid token limit issues
        const maxTextLength = 2000; // Limit each text segment to 2000 characters
        if (text.length > maxTextLength) {
            console.log(`🔄 FORMAT: Truncating text from ${text.length} to ${maxTextLength} characters`);
            text = text.substring(0, maxTextLength) + "... [truncated for length]";
        }
        
        // Format a more informative context entry
        if (text) {
            extractedTextCount++;
            console.log(`🔄 FORMAT: Extracted text length: ${text.length} characters`);
            
            const fileName = match.metadata?.file_name || 'Unknown';
            
            // Format with more structure and emphasis on dataset name
            let formattedEntry = `## Dataset: ${datasetName} (Relevance: ${match.score.toFixed(4)})\n`;
            if (organization) formattedEntry += `**Source/Organization**: ${organization}\n`;
            if (tags.length > 0) formattedEntry += `**Tags**: ${tags.join(', ')}\n`;
            if (match.id) formattedEntry += `**ID**: ${match.id}\n`;
            formattedEntry += `**File**: ${fileName}\n\n`;
            formattedEntry += `${text}\n\n`;
            formattedEntry += `${'='.repeat(80)}\n\n`;
            
            context += formattedEntry;
        } else {
            console.log('⚠️ FORMAT: No text could be extracted from this match');
        }
    });
    
    // Check if we had to limit the number of matches
    if (results.matches.length > maxMatches) {
        context += `Note: Showing ${maxMatches} most relevant results out of ${results.matches.length} total matches.\n\n`;
    }
    
    console.log(`✅ FORMAT: Formatting complete, extracted text from ${extractedTextCount}/${matchesToProcess.length} processed matches`);
    console.log(`✅ FORMAT: Total context length: ${context.length} characters`);
    
    // Add instructions for more specific results
    context += "\nPlease provide specific dataset names, organizations, and other key details in your responses when available. Focus on being precise about what datasets contain the requested information.";
    
    return context;
}

// Simple helper function to search Pinecone - enhanced similarly to search_json_data in chat_with_data.py
async function searchPinecone(embedding, topK = 8) {  // Increased from 5 to 8 for more results
    console.log('🔍 PINECONE SEARCH: Starting search operation');
    console.log('🔍 PINECONE SEARCH: Using index:', PINECONE_INDEX);
    
    if (!pineconeIndex) {
        console.error('❌ PINECONE ERROR: Index not initialized');
        return { matches: [] };
    }
    
    try {
        console.log(`📤 PINECONE SEARCH: Query embedding dimension: ${embedding.length}`);
        console.log(`📤 PINECONE SEARCH: Top K value: ${topK}`);
        
        // Log the first few values of the embedding for verification
        console.log(`📤 PINECONE SEARCH: Embedding sample: [${embedding.slice(0, 5).map(x => x.toFixed(6)).join(', ')}...]`);
        
        console.log('📤 PINECONE SEARCH: Sending query to Pinecone...');
        
        // Execute query with parameters matching chat_with_data.py but with improved parameters
        const queryStartTime = Date.now();
        
        // Create query object with possible improvements
        const queryParams = {
            vector: embedding,
            topK: topK,
            includeMetadata: true,
            includeValues: false,  // We don't need vector values in the response
        };
        
        // Add minimum score threshold to filter out low-relevance matches
        // This helps ensure we only get datasets that are actually relevant
        queryParams.scoreThreshold = 0.65;  // Only include results with >65% similarity
        
        console.log('📤 PINECONE SEARCH: Query parameters:', JSON.stringify(queryParams, null, 2));
        
        const response = await pineconeIndex.query(queryParams);
        const queryTime = Date.now() - queryStartTime;
        
        console.log(`✅ PINECONE SUCCESS: Query completed in ${queryTime}ms`);
        console.log(`✅ PINECONE SUCCESS: Found ${response.matches?.length || 0} matches`);
        
        // Log details about each match
        if (response.matches && response.matches.length > 0) {
            console.log('📝 PINECONE RESULTS: Match details:');
            response.matches.forEach((match, i) => {
                console.log(`  Match ${i+1}:`);
                console.log(`    ID: ${match.id}`);
                console.log(`    Score: ${match.score.toFixed(4)}`);
                
                // Try to identify key metadata fields
                const datasetName = match.metadata?.dataset_title || match.metadata?.title || 'Unknown';
                console.log(`    Dataset: ${datasetName}`);
                
                console.log(`    Metadata keys: ${Object.keys(match.metadata || {}).join(', ')}`);
                
                // Log file name if available
                if (match.metadata && match.metadata.file_name) {
                    console.log(`    File: ${match.metadata.file_name}`);
                }
            });
        } else {
            console.log('📝 PINECONE RESULTS: No matches found');
            
            // If no matches found, try a second search with lower threshold
            if (queryParams.scoreThreshold) {
                console.log('🔄 PINECONE SEARCH: Trying again with lower threshold...');
                delete queryParams.scoreThreshold;
                
                const fallbackResponse = await pineconeIndex.query(queryParams);
                
                console.log(`✅ PINECONE SUCCESS: Second query found ${fallbackResponse.matches?.length || 0} matches`);
                
                // Use these results instead
                return fallbackResponse;
            }
        }
        
        return response;
    } catch (error) {
        console.error('❌ PINECONE ERROR: Failed to search:', error);
        console.error('❌ PINECONE ERROR: Error name:', error.name);
        console.error('❌ PINECONE ERROR: Error message:', error.message);
        if (error.cause) {
            console.error('❌ PINECONE ERROR: Error cause:', error.cause);
        }
        
        // Try a fallback query without the score threshold if that was the problem
        try {
            console.log('🔄 PINECONE SEARCH: Attempting fallback query without additional parameters...');
            const fallbackResponse = await pineconeIndex.query({
                vector: embedding,
                topK: topK,
                includeMetadata: true
            });
            console.log(`✅ PINECONE SUCCESS: Fallback query found ${fallbackResponse.matches?.length || 0} matches`);
            return fallbackResponse;
        } catch (fallbackError) {
            console.error('❌ PINECONE ERROR: Fallback query also failed:', fallbackError.message);
            return { matches: [] };
        }
    }
}

// Chat API endpoint - updated to match chat_with_data functionality
app.post('/api/chat', async (req, res) => {
    try {
        console.log('🔄 API: Received chat request');
        
        const { query, useEmbeddings = true } = req.body;
        console.log('🔄 API: Query:', query);
        console.log('🔄 API: Client requested embeddings:', useEmbeddings);

        // Validate required fields
        if (!query) {
            console.log('❌ API: Missing query parameter');
            return res.status(400).json({ error: 'Missing query parameter' });
        }
        
        // Force embeddings to be used if Pinecone is available, regardless of client setting
        const shouldUseEmbeddings = usePinecone; // Always use if available
        console.log('🔄 API: Will use Pinecone embeddings:', shouldUseEmbeddings, '(forced to true if available)');
        
        let context = '';
        
        // If using embeddings, search Pinecone
        if (shouldUseEmbeddings) {
            try {
                console.log('🔄 API: Starting embedding and Pinecone search process');
                
                // Generate embedding for query
                console.log('🔄 API: Generating embedding from OpenAI');
                const embeddingStartTime = Date.now();
                const embedding = await generateEmbedding(query);
                const embeddingTime = Date.now() - embeddingStartTime;
                console.log(`✅ API: Generated embedding in ${embeddingTime}ms`);
                
                // Search Pinecone
                console.log('🔄 API: Searching Pinecone with embedding');
                const searchStartTime = Date.now();
                const searchResults = await searchPinecone(embedding);
                const searchTime = Date.now() - searchStartTime;
                console.log(`✅ API: Pinecone search completed in ${searchTime}ms`);
                console.log(`✅ API: Found ${searchResults.matches ? searchResults.matches.length : 0} matches`);
                
                // Format context using the new helper function
                console.log('🔄 API: Formatting context from search results');
                context = formatContextForChat(searchResults);
                console.log('✅ API: Context formatted successfully');
                console.log(`📝 API: Context length: ${context.length} characters`);
                
            } catch (err) {
                console.error('❌ API ERROR: Failed during embeddings/search process:', err);
                console.log('⚠️ API: Falling back to direct chat without context');
                // Continue without embeddings
            }
        }

        // Prepare the chat prompt - similar to the messages in chat_with_data.py
        const messages = [
            {
                role: 'system',
                content: shouldUseEmbeddings 
                    ? `You are a helpful assistant that provides detailed information about Virginia data. 
                    
IMPORTANT INSTRUCTIONS:
1. When answering, always reference specific dataset names, organizations, and data sources from the provided context.
2. Highlight the exact dataset names in your responses.
3. Be precise about what information comes from which dataset.
4. If multiple datasets contain relevant information, mention each one separately.
5. If the context doesn't contain enough information to answer the question, say so.
6. Do not make up dataset names or information that isn't in the context.

Your goal is to provide the most accurate and specific information about Virginia datasets.`
                    : `You are a helpful assistant for the Virginia Data Portal. When possible, refer to specific datasets and sources in your answers. Be specific about what information is available rather than providing general answers. If you don't know something specific, say so while being helpful and friendly.`
            }
        ];
        
        // Add context if available
        if (shouldUseEmbeddings && context) {
            messages.push({
                role: 'user', 
                content: `Context:\n${context}\n\nQuestion: ${query}`
            });
        } else {
            messages.push({
                role: 'user',
                content: query
            });
        }

        // Generate response using OpenAI
        console.log(`🔄 API: Generating response with OpenAI using model: ${CHAT_MODEL}...`);
        console.log('🔄 API: Preparing messages for OpenAI');
        
        // Log the message structure we're sending to OpenAI
        console.log(`🔄 API: System message length: ${messages[0].content.length} characters`);
        console.log(`🔄 API: User message count: ${messages.length - 1}`);
        
        if (shouldUseEmbeddings && context) {
            console.log(`🔄 API: Context + query length: ${messages[1].content.length} characters`);
            // Estimate token count (rough approximation: ~4 chars per token)
            const estimatedTokens = Math.ceil((messages[0].content.length + messages[1].content.length) / 4);
            console.log(`🔄 API: Estimated token count: ~${estimatedTokens} tokens`);
            
            // Check if we might exceed context limits
            if (estimatedTokens > 15000) {
                console.log('⚠️ API: Large context detected, using more aggressive truncation');
                
                // Create a shorter version of the context by further limiting results
                // We'll truncate the context to roughly half its size
                const shorterContext = context.split('='.repeat(80))[0] + 
                    '\n\n... [Additional results omitted for length] ...\n\n' + 
                    'Please provide specific dataset names and details from the available context.';
                
                // Replace the original context with the shortened version
                messages[1].content = `Context:\n${shorterContext}\n\nQuestion: ${query}`;
                console.log(`🔄 API: Shortened context + query length: ${messages[1].content.length} characters`);
            }
        }
        
        let chatResponse;
        try {
            console.log(`🔄 API: Sending request to OpenAI ChatGPT API (${CHAT_MODEL})`);
            const chatStartTime = Date.now();
            
            // First attempt with preferred model
            try {
                chatResponse = await openai.chat.completions.create({
                    model: CHAT_MODEL,
                    messages: messages,
                    temperature: 0.2,  // Lower temperature for more precise answers
                    max_tokens: 1500   // Increased max tokens for more detailed responses
                });
            } catch (modelError) {
                // If we hit a context length error, try with a model with larger context
                if (modelError.code === 'context_length_exceeded') {
                    console.log('⚠️ API: Context length exceeded with primary model, trying gpt-4-turbo');
                    
                    // Further reduce context if needed
                    if (shouldUseEmbeddings && context) {
                        // Take just the first dataset description
                        const firstDataset = context.split('='.repeat(80))[0] + 
                            '\n\n... [Additional results omitted for length] ...\n\n' + 
                            'Please answer based on the available context, noting that some results were omitted.';
                        
                        messages[1].content = `Context:\n${firstDataset}\n\nQuestion: ${query}`;
                    }
                    
                    // Try with gpt-4-turbo which has 128k context
                    chatResponse = await openai.chat.completions.create({
                        model: "gpt-4-turbo",
                        messages: messages,
                        temperature: 0.2,
                        max_tokens: 1500
                    });
                    console.log('✅ API: Successfully used fallback model gpt-4-turbo');
                } else {
                    // Rethrow any other errors
                    throw modelError;
                }
            }
            
            const chatTime = Date.now() - chatStartTime;
            console.log(`✅ API: Response generated successfully in ${chatTime}ms`);
            console.log(`✅ API: Response length: ${chatResponse.choices[0].message.content.length} characters`);
            
            // Log model usage information
            if (chatResponse.usage) {
                console.log('📊 API: Token usage:');
                console.log(`   - Prompt tokens: ${chatResponse.usage.prompt_tokens}`);
                console.log(`   - Completion tokens: ${chatResponse.usage.completion_tokens}`);
                console.log(`   - Total tokens: ${chatResponse.usage.total_tokens}`);
            }
        } catch (err) {
            console.error('❌ API ERROR: OpenAI API request failed:', err);
            console.error('❌ API ERROR: Error details:', err.message);
            return res.status(500).json({ 
                error: 'Failed to generate response with OpenAI', 
                details: err.message 
            });
        }

        // Send the response back to the client
        console.log('✅ API: Sending response to client');
        res.json({
            response: chatResponse.choices[0].message.content,
            usedEmbeddings: shouldUseEmbeddings && context !== 'No relevant data found.'
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
        console.log(`- Pinecone Host: ${PINECONE_HOST}`);
    }
}); 