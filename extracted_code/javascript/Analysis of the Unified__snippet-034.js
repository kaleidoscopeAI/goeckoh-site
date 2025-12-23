const request: OllamaRequest = {
  model: 'llama2',
  prompt,
  stream: false,
  options: {
    temperature: 0.7,
    top_p: 0.9,
    num_predict: 150,
    ...options
  }
};

return this.enqueueRequest(request);
