// DOM Elements
const messagesContainer = document.getElementById('messagesContainer');
const userInput = document.getElementById('userInput');
const sendButton = document.getElementById('sendButton');
const settingsButton = document.getElementById('settingsButton');
const checkIndexesButton = document.getElementById('checkIndexesButton');
const settingsModal = document.getElementById('settingsModal');
const closeSettings = document.getElementById('closeSettings');
const saveSettings = document.getElementById('saveSettings');
const useEmbeddingsToggle = document.getElementById('useEmbeddingsToggle');

// Application state - ALWAYS use embeddings by default (override any stored setting)
let useEmbeddings = true;
localStorage.setItem('useEmbeddings', 'true'); // Force it to true in storage

console.log('Initializing application...');

// Add a message to the chat
function addMessage(content, sender, metadata) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;
    
    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    
    // Format dataset results for better display
    if (sender === 'assistant' && (content.includes('**') || content.includes('## '))) {
        // This appears to be a dataset results message, format it nicely
        
        // Process markdown-style content
        let formattedContent = content;
        
        // Format headers (## Title)
        formattedContent = formattedContent.replace(/## ([^\n]+)/g, '<h3>$1</h3>');
        
        // Format numbered lists (1., 2., etc.)
        formattedContent = formattedContent.replace(/(\d+)\.\s+/g, '<strong>$1.</strong> ');
        
        // Format list items with bullets
        formattedContent = formattedContent.replace(/- /g, '• ');
        
        // Format dataset info lines (bold text followed by colon)
        formattedContent = formattedContent.replace(/\*\*([^*:]+):\*\*/g, '<strong>$1:</strong>');
        
        // Format regular bold text (that's not already handled)
        formattedContent = formattedContent.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
        
        // Format links - [text](url)
        formattedContent = formattedContent.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
        
        // Replace newlines with proper breaks
        formattedContent = formattedContent.replace(/\n\n/g, '</p><p>');
        formattedContent = formattedContent.replace(/\n/g, '<br>');
        
        // Wrap the content in a paragraph
        formattedContent = `<p>${formattedContent}</p>`;
        
        // Add a dataset result class for custom styling
        messageContent.classList.add('dataset-results');
        
        // Use innerHTML to render the formatted HTML
        messageContent.innerHTML = formattedContent;
    } else {
        // For regular text, use a simpler approach to handle bold markdown in regular messages
        // Split by newlines 
        const paragraphs = content.split('\n').filter(line => line.trim() !== '');
        
        paragraphs.forEach(paragraph => {
            const p = document.createElement('p');
            
            // Format bold text with regex
            let formattedText = paragraph;
            
            // Replace **text** with <strong>text</strong>
            formattedText = formattedText.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
            
            // Use innerHTML to render the formatted text
            p.innerHTML = formattedText;
            
            messageContent.appendChild(p);
        });
    }
    
    // Add metadata indicator if embeddings were used
    if (metadata && metadata.usedEmbeddings) {
        const metadataBadge = document.createElement('div');
        metadataBadge.className = 'metadata-badge';
        metadataBadge.innerHTML = '<i class="fas fa-database"></i> Using Virginia data knowledge base';
        messageContent.appendChild(metadataBadge);
    }
    
    messageDiv.appendChild(messageContent);
    messagesContainer.appendChild(messageDiv);
    
    // Scroll to bottom
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// Check server connection
async function checkServerStatus() {
    addMessage('Checking server connection...', 'system');
    console.log('Checking server connection...');
    
    try {
        const response = await fetch('/api/diagnostics');
        
        if (!response.ok) {
            throw new Error('Server diagnostics failed');
        }
        
        const data = await response.json();
        console.log('Server diagnostics data:', data);
        
        // Build status message
        let statusMessage = '';
        
        if (data.openai) {
            statusMessage += '✅ Connected to OpenAI successfully\n';
            console.log('Connected to OpenAI successfully');
            if (data.openaiModels) {
                // statusMessage += `Available models: ${data.openaiModels.join(', ')}\n\n`;
                console.log('Available OpenAI models:', data.openaiModels);
            }
        } else {
            statusMessage += '❌ Failed to connect to OpenAI: ' + (data.openaiError || 'Unknown error') + '\n\n';
            console.error('Failed to connect to OpenAI:', data.openaiError || 'Unknown error');
        }
        
        if (data.usePinecone) {
            if (data.pinecone) {
                statusMessage += '✅ Connected to Pinecone successfully\n';
                console.log('Connected to Pinecone successfully');
                
                if (data.directConnectionSuccess) {
                    statusMessage += `✅ Successfully connected to index at ${data.pineconeIndexHost}\n`;
                    console.log('Successfully connected to Pinecone index at', data.pineconeIndexHost);
                    if (data.indexStats) {
                        statusMessage += `Total vectors: ${data.indexStats.totalVectorCount || 'unknown'}\n`;
                        statusMessage += `Dimension: ${data.indexStats.dimension || 'unknown'}\n`;
                        console.log('Pinecone index stats:', data.indexStats);
                    }
                } else if (data.directConnectionError) {
                    statusMessage += `❌ Failed to connect to direct index: ${data.directConnectionError}\n`;
                    console.error('Failed to connect to direct Pinecone index:', data.directConnectionError);
                }
                
                if (data.pineconeIndexes && data.pineconeIndexes.length > 0) {
                    statusMessage += '\nAvailable indexes:\n';
                    data.pineconeIndexes.forEach(idx => {
                        statusMessage += `- ${idx.name}${idx.direct ? ' (direct)' : ''}\n`;
                        console.log('Available Pinecone index:', idx.name);
                    });
                } else {
                    statusMessage += 'No indexes found in your Pinecone account.\n';
                    console.warn('No indexes found in Pinecone account');
                }
            } else {
                statusMessage += '❌ Failed to connect to Pinecone: ' + (data.pineconeError || 'Unknown error');
                console.error('Failed to connect to Pinecone:', data.pineconeError || 'Unknown error');
            }
        } else {
            statusMessage += '⚠️ Pinecone integration is disabled. Only direct chat is available.';
            console.warn('Pinecone integration is disabled');
        }
        
        addMessage(statusMessage, 'system');
        
        // Update UI based on Pinecone availability
        if (useEmbeddingsToggle) {
            if (data.usePinecone && data.pinecone && (data.directConnectionSuccess || data.indexExists)) {
                useEmbeddingsToggle.disabled = false;
                useEmbeddingsToggle.parentElement.classList.remove('disabled');
            } else {
                useEmbeddingsToggle.disabled = true;
                useEmbeddingsToggle.checked = false;
                useEmbeddings = false;
                localStorage.setItem('useEmbeddings', 'false');
                useEmbeddingsToggle.parentElement.classList.add('disabled');
            }
        }
        
    } catch (error) {
        console.error('Error checking server status:', error);
        addMessage('Failed to check server status: ' + error.message, 'system');
    }
}

// Handle user input
async function handleUserInput() {
    const message = userInput.value.trim();
    if (!message) return;
    
    // Add user message to chat
    addMessage(message, 'user');
    userInput.value = '';
    userInput.style.height = 'auto';
    
    // Disable send button and show loading state
    sendButton.disabled = true;
    sendButton.innerHTML = '<span class="loading"></span>';
    
    try {
        const response = await fetchChatResponse(message);
        addMessage(response.response, 'assistant', { usedEmbeddings: response.usedEmbeddings });
    } catch (error) {
        console.error('Chat API error:', error);
        addMessage('Sorry, there was an error processing your request. Please try again later.', 'system');
    } finally {
        // Re-enable send button
        sendButton.disabled = false;
        sendButton.innerHTML = '<i class="fas fa-paper-plane"></i>';
    }
}

// Fetch chat response from the server
async function fetchChatResponse(message) {
    try {
        // Prepare the request body - always use embeddings regardless of toggle state
        const requestBody = {
            query: message,
            useEmbeddings: true, // Always set to true to force using the vector db
            systemPrompt: "You are a data scientist assistant specializing in Virginia data. Use the provided context to help users explore datasets, suggest relevant datasets for projects, and assist in building predictive models.",
            requestType: "listDatasets" // Example request type, can be changed based on user input
        };
        
        console.log("Sending request with embeddings:", requestBody.useEmbeddings);
        
        // Call the backend chat API
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestBody)
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            const errorMessage = errorData.error || 'Unknown error';
            throw new Error(`Server error: ${errorMessage}`);
        }
        
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('API request failed:', error);
        throw error;
    }
}

// Toggle embeddings usage - no longer actually changes the behavior
function toggleEmbeddings(event) {
    // Force it to true regardless of what the user selects
    useEmbeddings = true;
    event.target.checked = true;
    localStorage.setItem('useEmbeddings', 'true');
    console.log('Knowledge base always enabled for better results');
}

// Event Listeners
sendButton.addEventListener('click', handleUserInput);

userInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleUserInput();
    }
});

// Auto-resize textarea as user types
userInput.addEventListener('input', () => {
    userInput.style.height = 'auto';
    userInput.style.height = (userInput.scrollHeight) + 'px';
});

// Check connection button
checkIndexesButton.addEventListener('click', checkServerStatus);

// Settings modal
settingsButton.addEventListener('click', () => {
    settingsModal.style.display = 'flex';
});

closeSettings.addEventListener('click', () => {
    settingsModal.style.display = 'none';
});

// Close modal when clicking outside
settingsModal.addEventListener('click', (e) => {
    if (e.target === settingsModal) {
        settingsModal.style.display = 'none';
    }
});

// Toggle embeddings if the element exists
if (useEmbeddingsToggle) {
    useEmbeddingsToggle.checked = useEmbeddings;
    useEmbeddingsToggle.addEventListener('change', toggleEmbeddings);
}

// Initialize app
// document.addEventListener('DOMContentLoaded', () => {
//     // Show welcome message
//     addMessage('Welcome to the Virginia Data Portal Chat! As your data scientist assistant, I am equipped to help you explore datasets from the Virginia Open Data Portal, suggest relevant datasets for your projects, and assist in building predictive models. Feel free to ask any questions related to data analysis, model building, or dataset exploration.', 'system');
    
//     // Check server status on load
//     setTimeout(() => {
//         checkServerStatus();
//     }, 1000);
// });

function showExampleQuestions() {
    const examples = `Here are some example questions you can ask:

1. What datasets are available for bridges and roads?
2. Can you suggest datasets for building a predictive model on traffic patterns?
3. How can I use the datasets to analyze economic trends in Virginia?
4. Can you help me find datasets related to environmental studies?
5. What datasets are available for educational statistics in Virginia?
6. What datasets have traffic records?
7. What datasets have traffic accidents?
8. What datasets have placemaking data?`;
    addMessage(examples, 'system');
}

// Call the function to show example questions after checking server status
setTimeout(() => {
    checkServerStatus();
    showExampleQuestions();
}, 1000); 