async function generateOllamaCompletion(promptText, modelName = "llama3.1") {
  const ollamaEndpoint = 'http://localhost:11434/api/generate';

  const requestBody = {
    model: modelName,
    prompt: promptText,
    stream: false // Request a single JSON response object
    // Add other options like 'format', 'options', 'keep_alive' as needed
  };

  try {
    const response = await fetch(ollamaEndpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      // Handle HTTP errors (e.g., 404, 500)
      const errorText = await response.text();
      throw new Error(`HTTP error ${response.status}: ${errorText}`);
    }

    const responseData = await response.json();
    // Process the responseData, e.g., responseData.response contains the generated text
    console.log("Generated Response:", responseData.response);
    return responseData;

  } catch (error) {
    console.error("Error calling Ollama generate API:", error);
    // Handle fetch errors (network issues, etc.) or JSON parsing errors
    throw error;
  }
}

// Example usage:
// generateOllamaCompletion("Explain the Ornstein-Uhlenbeck process briefly.");

async function chatWithOllama(messages, modelName = "llama3.1") {
  const ollamaEndpoint = 'http://localhost:11434/api/chat';

  // Ensure messages is an array of { role: '...', content: '...' } objects
  if (!Array.isArray(messages) || messages.length === 0) {
      throw new Error("Messages array cannot be empty.");
  }

  const requestBody = {
    model: modelName,
    messages: messages,
    stream: false // Request a single JSON response object
    // Add other options like 'format', 'options', 'keep_alive', 'tools' as needed
  };

  try {
    const response = await fetch(ollamaEndpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`HTTP error ${response.status}: ${errorText}`);
    }

    const responseData = await response.json();
    // Process the responseData, e.g., responseData.message.content
    console.log("Chat Response:", responseData.message.content);
    return responseData;

  } catch (error) {
    console.error("Error calling Ollama chat API:", error);
    throw error;
  }
}

// Example usage:
/*
const chatHistory = [
  { role: 'system', content: 'You are a helpful assistant.' },
  { role: 'user', content: 'What is the capital of France?' }
];
chatWithOllama(chatHistory).then(response => {
  // Add assistant's response to history for next turn
  chatHistory.push(response.message);
  //... continue conversation...
});
*/