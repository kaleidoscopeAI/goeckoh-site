const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('goeckohDesktop', {
  activateLicense: (key) => ipcRenderer.invoke('activate-license', key),
  onActivationError: (cb) => ipcRenderer.on('activation-error', (_evt, msg) => cb(msg)),
  logMetric: (metric) => ipcRenderer.send('log-metric', metric),
});
