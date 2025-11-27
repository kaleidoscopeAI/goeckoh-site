import { app, BrowserWindow, nativeImage } from 'electron';
import path from 'node:path';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';
import { setTimeout as delay } from 'node:timers/promises';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const devUrl = process.env.VITE_DEV_SERVER_URL;
const distIndex = path.join(__dirname, 'dist', 'index.html');
const iconPath = path.join(__dirname, 'icons', 'app.png');
const OLLAMA_URL = process.env.VITE_OLLAMA_URL || 'http://localhost:11434';
const SD_HOST = process.env.SD_URL || process.env.VITE_SD_URL || 'http://localhost:7860';
const SD_WEBUI_CMD = process.env.SD_WEBUI_CMD; // optional: command to auto-start A1111

function createWindow() {
  const windowIcon = fs.existsSync(iconPath) ? nativeImage.createFromPath(iconPath) : undefined;

  const win = new BrowserWindow({
    width: 1280,
    height: 800,
    backgroundColor: '#000000',
    icon: windowIcon,
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  win.setMenu(null);

  if (devUrl) {
    win.loadURL(devUrl);
  } else if (fs.existsSync(distIndex)) {
    win.loadFile(distIndex);
  } else {
    const message = `
      <style>
        body { margin: 0; padding: 48px; background: #0a0a0a; color: #eee; font-family: Arial, sans-serif; }
        h1 { margin-bottom: 12px; }
        code { background: #111; padding: 2px 6px; border-radius: 4px; }
      </style>
      <h1>Cognitive Nebula</h1>
      <p>Build output not found. Run <code>npm run build</code> first, then <code>npm run desktop</code>.</p>
      <p>For live dev, start the Vite server and launch Electron with <code>VITE_DEV_SERVER_URL=http://localhost:5173 electron .</code></p>
    `;
    win.loadURL(`data:text/html;charset=utf-8,${encodeURIComponent(message)}`);
  }
}

async function waitForService(url, path, timeoutMs = 15000, intervalMs = 750) {
  const started = Date.now();
  while (Date.now() - started < timeoutMs) {
    try {
      const res = await fetch(`${url}${path}`, { method: 'GET' });
      if (res.ok) return true;
    } catch (err) {
      // Ignore until timeout
    }
    await delay(intervalMs);
  }
  return false;
}

async function ensureOllama() {
  const available = await waitForService(OLLAMA_URL, '/api/version', 1000, 500);
  if (available) {
    console.log(`[ollama] detected at ${OLLAMA_URL}`);
    return;
  }

  try {
    console.log('[ollama] starting local serve...');
    const child = spawn('ollama', ['serve'], {
      detached: true,
      stdio: 'ignore',
    });
    child.unref();
  } catch (err) {
    console.warn('[ollama] could not start automatically. Ensure Ollama is installed and running.', err);
    return;
  }

  const ready = await waitForService(OLLAMA_URL, '/api/version', 15000, 1000);
  if (ready) {
    console.log(`[ollama] ready at ${OLLAMA_URL}`);
  } else {
    console.warn(`[ollama] not reachable at ${OLLAMA_URL}. Start it manually if the app cannot generate responses.`);
  }
}

async function ensureAutomatic1111() {
  const available = await waitForService(SD_HOST, '/sdapi/v1/options', 1000, 500);
  if (available) {
    console.log(`[automatic1111] detected at ${SD_HOST}`);
    return;
  }

  if (!SD_WEBUI_CMD) {
    console.warn(`[automatic1111] not reachable at ${SD_HOST}. Set SD_WEBUI_CMD to auto-start, or start it manually for local image generation.`);
    return;
  }

  try {
    console.log(`[automatic1111] starting with command: ${SD_WEBUI_CMD}`);
    const child = spawn(SD_WEBUI_CMD, {
      shell: true,
      detached: true,
      stdio: 'ignore',
    });
    child.unref();
  } catch (err) {
    console.warn('[automatic1111] could not start automatically. Ensure it is installed and running.', err);
    return;
  }

  const ready = await waitForService(SD_HOST, '/sdapi/v1/options', 20000, 1000);
  if (ready) {
    console.log(`[automatic1111] ready at ${SD_HOST}`);
  } else {
    console.warn(`[automatic1111] still unreachable at ${SD_HOST}. Start it manually if image generation fails.`);
  }
}

app.whenReady().then(() => {
  // Attempt to ensure the local AI runtime is running so the renderer works after clicking the icon.
  ensureOllama();
  ensureAutomatic1111();

  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});
