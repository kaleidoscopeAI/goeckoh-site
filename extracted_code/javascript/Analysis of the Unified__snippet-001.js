// scripts/bundle-ollama.js
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { spawn } from 'child_process';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const OLLAMA_DIR = path.join(__dirname, '../ollama-binaries');

// Create directory for Ollama binaries
if (!fs.existsSync(OLLAMA_DIR)) {
  fs.mkdirSync(OLLAMA_DIR, { recursive: true });
}

// Platform-specific Ollama download URLs
const OLLAMA_BINARIES = {
  darwin: {
    x64: 'https://github.com/jmorganca/ollama/releases/download/v0.1.0/ollama-darwin',
    arm64: 'https://github.com/jmorganca/ollama/releases/download/v0.1.0/ollama-darwin-arm64'
  },
  win32: {
    x64: 'https://github.com/jmorganca/ollama/releases/download/v0.1.0/ollama-windows-amd64.exe'
  },
  linux: {
    x64: 'https://github.com/jmorganca/ollama/releases/download/v0.1.0/ollama-linux-amd64'
  }
};

async function downloadOllama() {
  const platform = process.platform;
  const arch = process.arch;
  
  console.log(`📦 Bundling Ollama for ${platform}-${arch}`);
  
  const platformUrls = OLLAMA_BINARIES[platform];
  if (!platformUrls) {
    console.warn(`Unsupported platform: ${platform}`);
    return;
  }
  
  const url = platformUrls[arch];
  if (!url) {
    console.warn(`Unsupported architecture: ${arch} for platform ${platform}`);
    return;
  }
  
  const binaryName = platform === 'win32' ? 'ollama.exe' : 'ollama';
  const binaryPath = path.join(OLLAMA_DIR, binaryName);
  
  // Download Ollama binary
  console.log(`Downloading Ollama from: ${url}`);
  
  try {
    const response = await fetch(url);
    const buffer = await response.arrayBuffer();
    fs.writeFileSync(binaryPath, Buffer.from(buffer));
    
    // Make executable on Unix systems
    if (platform !== 'win32') {
      fs.chmodSync(binaryPath, '755');
    }
    
    console.log(`✅ Ollama bundled successfully: ${binaryPath}`);
  } catch (error) {
    console.error('Failed to download Ollama:', error);
  }
}

// Run the bundling
downloadOllama().catch(console.error);
