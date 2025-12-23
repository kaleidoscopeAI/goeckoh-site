try {
  await this.ollamaEngine.start();
  await this.ollamaEngine.ensureModel('llama2');
  console.log('✅ Embedded Ollama engine ready');
} catch (error) {
  console.warn('❌ Embedded Ollama failed, using cognitive fallback:', error);
}
