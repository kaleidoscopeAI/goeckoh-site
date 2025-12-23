return new Promise((resolve, reject) => {
  const pullProcess = spawn(this.OLLAMA_BINARY, ['pull', model]);

  pullProcess.stdout?.on('data', (data) => {
    console.log('Model Pull:', data.toString());
  });

  pullProcess.stderr?.on('data', (data) => {
    console.error('Pull Error:', data.toString());
  });

  pullProcess.on('close', (code) => {
    if (code === 0) {
      resolve();
    } else {
      reject(new Error(`Model pull failed with code ${code}`));
    }
  });
});
