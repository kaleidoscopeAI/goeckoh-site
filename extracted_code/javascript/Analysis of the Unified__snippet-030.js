const platform = process.platform;
const arch = process.arch;

// In production, you'd bundle the Ollama binary with your app
const binaryPaths = {
  darwin: {
    x64: '/Applications/Ollama.app/Contents/Resources/ollama',
    arm64: '/Applications/Ollama.app/Contents/Resources/ollama'
  },
  win32: {
    x64: 'C:\\Program Files\\Ollama\\ollama.exe',
    ia32: 'C:\\Program Files\\Ollama\\ollama.exe'
  },
  linux: {
    x64: '/usr/local/bin/ollama',
    arm64: '/usr/local/bin/ollama'
  }
};

const platformPaths = binaryPaths[platform as keyof typeof binaryPaths];
if (platformPaths) {
  const binaryPath = platformPaths[arch as keyof typeof platformPaths];
  if (fs.existsSync(binaryPath)) {
    return binaryPath;
  }
}

// Fallback to system PATH
return 'ollama';
