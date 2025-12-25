const { contextBridge, ipcRenderer } = require('electron');

// Expose protected methods that allow the renderer process to use
// the ipcRenderer without exposing the entire object
contextBridge.exposeInMainWorld(
  'electronAPI', {
    onPythonLog: (callback) => ipcRenderer.on('python-log', callback),
    onPythonError: (callback) => ipcRenderer.on('python-error', callback),
    startSystem: () => ipcRenderer.send('start-system'),
    stopSystem: () => ipcRenderer.send('stop-system'),
    removeAllListeners: (channel) => ipcRenderer.removeAllListeners(channel)
  }
);
