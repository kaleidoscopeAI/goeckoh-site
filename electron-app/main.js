const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

let mainWindow;
let pythonProcess;

function findPython() {
  // Try to find Python in common locations
  const pythonCommands = ['python3', 'python', 'py'];
  
  for (const cmd of pythonCommands) {
    try {
      const result = require('child_process').spawnSync(cmd, ['--version'], { encoding: 'utf-8' });
      if (result.status === 0) {
        return cmd;
      }
    } catch (e) {
      continue;
    }
  }
  
  return 'python3'; // Default fallback
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false
    },
    icon: path.join(__dirname, 'build', 'icon.png')
  });

  // Load the UI
  mainWindow.loadFile('index.html');

  // Open DevTools in development
  if (process.env.NODE_ENV === 'development') {
    mainWindow.webContents.openDevTools();
  }

  mainWindow.on('closed', function () {
    mainWindow = null;
  });
}

function startPythonBackend() {
  const pythonCmd = findPython();
  const scriptPath = app.isPackaged
    ? path.join(process.resourcesPath, 'desktop_app.py')
    : path.join(__dirname, '..', 'desktop_app.py');

  console.log('Starting Python backend:', pythonCmd, scriptPath);

  pythonProcess = spawn(pythonCmd, [scriptPath, '--mode', 'child'], {
    stdio: ['pipe', 'pipe', 'pipe'],
    cwd: app.isPackaged 
      ? process.resourcesPath 
      : path.join(__dirname, '..')
  });

  pythonProcess.stdout.on('data', (data) => {
    console.log(`Python: ${data}`);
    if (mainWindow) {
      mainWindow.webContents.send('python-log', data.toString());
    }
  });

  pythonProcess.stderr.on('data', (data) => {
    console.error(`Python Error: ${data}`);
    if (mainWindow) {
      mainWindow.webContents.send('python-error', data.toString());
    }
  });

  pythonProcess.on('close', (code) => {
    console.log(`Python process exited with code ${code}`);
  });
}

function stopPythonBackend() {
  if (pythonProcess) {
    pythonProcess.kill();
    pythonProcess = null;
  }
}

app.whenReady().then(() => {
  createWindow();
  startPythonBackend();

  app.on('activate', function () {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', function () {
  stopPythonBackend();
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('before-quit', () => {
  stopPythonBackend();
});

// IPC handlers
ipcMain.on('start-system', () => {
  console.log('Starting system...');
  // System is already started in background
});

ipcMain.on('stop-system', () => {
  console.log('Stopping system...');
  stopPythonBackend();
});
