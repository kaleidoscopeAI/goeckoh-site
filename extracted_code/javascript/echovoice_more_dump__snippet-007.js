async function chatWithOllama(messages, modelName = "llama3.1") {
  const ollamaEndpoint = 'http://localhost:11434/api/chat';

  // Ensure messages is an array of { role: '...', content: '...' } objects
  if (!Array.isArray(messages) |
