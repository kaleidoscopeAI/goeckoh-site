if (this.isRunning) return;

return new Promise((resolve, reject) => {
  try {
    // Start Ollama as a subprocess
    this.ollamaProcess = spawn(this.OLLAMA_BINARY, ['serve'], {
      stdio: ['pipe', 'pipe', 'pipe'],
      env: { ...process.env, OLLAMA_HOST: '127.0.0.1:11435' } // Use different port to avoid conflicts
    });

    this.ollamaProcess.stdout?.on('data', (data) => {
      const output = data.toString();
      console.log('🧠 Ollama:', output);
      if (output.includes('Listening')) {
        this.isRunning = true;
        resolve();
      }
    });

    this.ollamaProcess.stderr?.on('data', (data) => {
      console.error('Ollama Error:', data.toString());
    });

    this.ollamaProcess.on('error', (error) => {
      console.error('Failed to start Ollama:', error);
      reject(error);
    });

    this.ollamaProcess.on('exit', (code) => {
      console.log(`Ollama process exited with code ${code}`);
      this.isRunning = false;
      this.emit('stopped');
    });

    // Wait for startup with timeout
    setTimeout(() => {
      if (!this.isRunning) {
        reject(new Error('Ollama startup timeout'));
      }
    }, 10000);

  } catch (error) {
    reject(error);
  }
});
