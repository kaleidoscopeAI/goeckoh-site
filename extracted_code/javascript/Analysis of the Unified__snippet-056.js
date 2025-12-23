const response = await fetch(url);
const buffer = await response.arrayBuffer();
fs.writeFileSync(binaryPath, Buffer.from(buffer));

// Make executable on Unix systems
if (platform !== 'win32') {
  fs.chmodSync(binaryPath, '755');
}

console.log(`✅ Ollama bundled successfully: ${binaryPath}`);
