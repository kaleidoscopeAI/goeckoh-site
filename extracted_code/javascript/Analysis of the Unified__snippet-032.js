if (!this.isRunning) await this.start();

// Check if model exists, pull if not
try {
  await this.makeRequest('GET', '/api/tags');
} catch (error) {
  console.log(`Pulling model ${model}...`);
  await this.pullModel(model);
}
