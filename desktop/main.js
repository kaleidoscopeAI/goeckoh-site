// Goeckoh Desktop — Electron shell around the same correction engine that
// runs in the browser (desktop/app/index.html is a bundled copy of
// real_time_therapeutic_voice_cloning_system.html; its AudioWorklet DSP is
// unmodified). This file adds what the web version can't do on its own:
// real device-bound license activation, cached locally so the app keeps
// working with no internet connection after the first activation.

const { app, BrowserWindow, ipcMain, Tray, Menu, nativeImage, powerSaveBlocker } = require('electron');
const path = require('path');
const fs = require('fs');
const http = require('http');
const crypto = require('crypto');
const https = require('https');
const { startLocalBackend } = require('./local-backend');

let localBackend = null;

let tray = null;
let powerBlockerId = null;
let isQuitting = false;

const BACKEND_HOST = 'goeckoh-backend.fly.dev';
const LICENSE_FILE = () => path.join(app.getPath('userData'), 'license.json');
const DEVICE_ID_FILE = () => path.join(app.getPath('userData'), 'device-id');

function getOrCreateDeviceId() {
  const f = DEVICE_ID_FILE();
  if (fs.existsSync(f)) return fs.readFileSync(f, 'utf8').trim();
  const id = crypto.randomUUID();
  fs.mkdirSync(path.dirname(f), { recursive: true });
  fs.writeFileSync(f, id);
  return id;
}

function readCachedLicense() {
  try { return JSON.parse(fs.readFileSync(LICENSE_FILE(), 'utf8')); }
  catch (e) { return null; }
}

function writeCachedLicense(data) {
  fs.mkdirSync(path.dirname(LICENSE_FILE()), { recursive: true });
  fs.writeFileSync(LICENSE_FILE(), JSON.stringify(data, null, 2));
}

// Plain https POST (no extra deps) — used for both /license/activate and
// /license/validate, which have an identical request/response shape.
function postJson(pathName, body, timeoutMs = 8000) {
  return new Promise((resolve, reject) => {
    const data = JSON.stringify(body);
    const req = https.request(
      {
        hostname: BACKEND_HOST,
        path: pathName,
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(data) },
        timeout: timeoutMs,
      },
      (res) => {
        let raw = '';
        res.on('data', (c) => (raw += c));
        res.on('end', () => {
          let json;
          try { json = JSON.parse(raw); } catch (e) { json = { detail: raw }; }
          if (res.statusCode >= 200 && res.statusCode < 300) resolve(json);
          else reject(Object.assign(new Error(json.detail || `HTTP ${res.statusCode}`), { status: res.statusCode, body: json }));
        });
      }
    );
    req.on('timeout', () => req.destroy(new Error('Request timed out — offline?')));
    req.on('error', reject);
    req.write(data);
    req.end();
  });
}

// Only these exact backend-authored strings mean "we genuinely reached
// license.validate on our own server, and it made a real decision" — an
// unrecognized error body means some intermediary (corporate/school
// firewall, captive wifi portal, or — in this dev sandbox — the egress
// policy gateway) answered instead of the real backend, which must be
// treated as "unreachable," not as an authoritative rejection. Without
// this check, anyone on a restrictive network gets wrongly locked out of
// their own offline-cached license instead of falling back to it.
const KNOWN_BACKEND_REJECTIONS = new Set([
  'License key not found',
  'subscription_lapsed',
  'payment_pending',
  'grace_period_expired',
]);
function isGenuineBackendRejection(err) {
  const detail = err.body?.detail;
  if (typeof detail !== 'string') return false;
  if (KNOWN_BACKEND_REJECTIONS.has(detail)) return true;
  return detail.startsWith('Device limit reached');
}

async function activateLicense(licenseKey) {
  const device_fingerprint = getOrCreateDeviceId();
  const result = await postJson('/license/activate', {
    license_key: licenseKey.trim().toUpperCase(),
    device_fingerprint,
    platform: process.platform,
  });
  writeCachedLicense({
    license_key: licenseKey.trim().toUpperCase(),
    device_fingerprint,
    token: result.token,
    plan: result.plan,
    activated_at: Date.now(),
    last_validated_at: Date.now(),
  });
  return result;
}

// Called on every launch. Tries to refresh the token online (also lets the
// server enforce device-count limits and catch a lapsed subscription); if
// that fails for any reason (no internet — the actual "offline" case this
// whole app exists for), it falls back to the last-known-good cached
// license rather than locking the user out. Only a hard REVOKED/expired
// response from the server (i.e. we *did* reach it) blocks access.
async function ensureActivated() {
  const cached = readCachedLicense();
  if (!cached) return { activated: false };

  try {
    const result = await postJson('/license/validate', {
      license_key: cached.license_key,
      device_fingerprint: cached.device_fingerprint,
    }, 5000);
    writeCachedLicense({ ...cached, token: result.token, plan: result.plan, last_validated_at: Date.now() });
    return { activated: true, mode: 'online', plan: result.plan };
  } catch (e) {
    if (process.env.GOECKOH_DEBUG_LOG) {
      fs.appendFileSync(process.env.GOECKOH_DEBUG_LOG,
        `validate failed: message=${e.message} status=${e.status} code=${e.code} stack=${e.stack}\n`);
    }
    if (isGenuineBackendRejection(e)) {
      // We reached OUR server and it explicitly said no (revoked/expired/etc).
      return { activated: false, error: e.body?.detail || e.message };
    }
    // Unreachable, or some other network layer answered instead of our
    // backend — genuine offline case. Trust the cache.
    if (cached.token) return { activated: true, mode: 'offline', plan: cached.plan };
    return { activated: false, error: 'No cached license and backend unreachable.' };
  }
}

// Serves desktop/app/ over http://127.0.0.1 instead of file:// — gives the
// AudioWorklet Blob-URL module and relative asset paths a stable origin,
// which is more reliable across platforms than raw file:// loading.
function startLocalServer() {
  const root = path.join(__dirname, 'app');
  const mime = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.png': 'image/png', '.svg': 'image/svg+xml' };
  return new Promise((resolve) => {
    const server = http.createServer((req, res) => {
      let p = decodeURIComponent(req.url.split('?')[0]);
      if (p === '/') p = '/index.html';
      const full = path.join(root, p);
      if (!full.startsWith(root)) { res.writeHead(403); res.end(); return; }
      fs.readFile(full, (err, buf) => {
        if (err) { res.writeHead(404); res.end('Not found'); return; }
        res.writeHead(200, { 'Content-Type': mime[path.extname(full)] || 'application/octet-stream' });
        res.end(buf);
      });
    });
    server.listen(0, '127.0.0.1', () => resolve(server.address().port));
  });
}

let mainWindow;

async function createWindow() {
  if (process.env.GOECKOH_DEBUG_LOG) {
    fs.appendFileSync(process.env.GOECKOH_DEBUG_LOG,
      `userData=${app.getPath('userData')} licenseExists=${fs.existsSync(LICENSE_FILE())} argv=${JSON.stringify(process.argv)}\n`);
  }
  const port = await startLocalServer();
  try {
    localBackend = await startLocalBackend(app.getPath('userData'));
  } catch (e) {
    // Port 8000 already in use (e.g. a real local Python backend already
    // running there for dev/testing) — session logging just won't be
    // available this run rather than crashing the whole app over it.
    if (process.env.GOECKOH_DEBUG_LOG) {
      fs.appendFileSync(process.env.GOECKOH_DEBUG_LOG, `local backend on :8000 failed to start: ${e.message}\n`);
    }
  }

  const startHidden = process.argv.includes('--hidden');
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 860,
    show: !startHidden,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  const status = await ensureActivated();
  if (status.activated) {
    mainWindow.loadURL(`http://127.0.0.1:${port}/index.html`);
    // "Run effortlessly" means correction starts on its own, not behind a
    // click the user has to remember to make every launch — including when
    // auto-started hidden at login, where there's no one watching to click
    // anything. quickStart() is the page's own existing entry point
    // (unchanged), just triggered automatically instead of on click.
    mainWindow.webContents.once('did-finish-load', () => {
      mainWindow.webContents.executeJavaScript('typeof quickStart === "function" && quickStart();').catch(() => {});
    });
    // Correction only matters while the mic pipeline is actually running,
    // which is the entire time this app's real page is loaded — keep the
    // OS from idling/display-sleeping the app so it keeps processing while
    // minimized to tray. This does NOT override a closed laptop lid; most
    // OSes suspend on lid-close regardless of app-level power blockers
    // unless the system's own power settings say otherwise — that's an OS
    // setting, not something this app can force.
    if (powerBlockerId === null) powerBlockerId = powerSaveBlocker.start('prevent-app-suspension');
  } else {
    mainWindow.loadFile(path.join(__dirname, 'activation.html'));
    mainWindow.webContents.once('did-finish-load', () => {
      if (status.error) mainWindow.webContents.send('activation-error', status.error);
    });
  }

  // Closing the window hides it to the tray instead of quitting — the whole
  // point of running "effortlessly in the background" is that the mic
  // pipeline keeps running without a visible window. Only the tray menu's
  // "Quit Goeckoh" (or Cmd+Q on macOS) actually exits the process.
  mainWindow.on('close', (e) => {
    if (isQuitting) return;
    e.preventDefault();
    mainWindow.hide();
    if (process.platform === 'darwin') app.dock?.hide();
  });

  ipcMain.handle('activate-license', async (_evt, key) => {
    try {
      await activateLicense(key);
      mainWindow.loadURL(`http://127.0.0.1:${port}/index.html`);
      if (powerBlockerId === null) powerBlockerId = powerSaveBlocker.start('prevent-app-suspension');
      return { ok: true };
    } catch (e) {
      return { ok: false, error: e.body?.detail || e.message };
    }
  });

  ipcMain.on('log-metric', (_evt, metric) => {
    if (localBackend) {
      try { localBackend.appendMetric(metric); } catch (e) {}
    }
  });

  setupTray();
}

function setupTray() {
  if (tray) return;
  const iconPath = path.join(__dirname, 'app', 'images', 'logo-light.png');
  let icon = nativeImage.createFromPath(iconPath);
  if (!icon.isEmpty()) icon = icon.resize({ width: 20, height: 20 });
  tray = new Tray(icon.isEmpty() ? nativeImage.createEmpty() : icon);
  tray.setToolTip('Goeckoh — running in background');
  const rebuildMenu = () => {
    const loginSettings = app.getLoginItemSettings();
    const menu = Menu.buildFromTemplate([
      { label: 'Show Goeckoh', click: () => { mainWindow.show(); if (process.platform === 'darwin') app.dock?.show(); } },
      { type: 'separator' },
      {
        label: 'Start automatically at login',
        type: 'checkbox',
        checked: loginSettings.openAtLogin,
        click: (item) => {
          app.setLoginItemSettings({ openAtLogin: item.checked, openAsHidden: true, args: ['--hidden'] });
        },
      },
      { type: 'separator' },
      {
        label: 'Quit Goeckoh',
        click: () => {
          isQuitting = true;
          app.quit();
        },
      },
    ]);
    tray.setContextMenu(menu);
  };
  rebuildMenu();
  tray.on('click', () => { mainWindow.show(); if (process.platform === 'darwin') app.dock?.show(); });
}

// Electron denies getUserMedia by default unless a handler explicitly grants
// it — without this, the entire product (which is nothing without mic
// access) would silently fail on first use for every real user. Scoped to
// only the app's own local origin and only microphone, not a blanket grant.
app.whenReady().then(() => {
  const { session } = require('electron');
  session.defaultSession.setPermissionRequestHandler((webContents, permission, callback) => {
    const url = webContents.getURL();
    const isOwnApp = url.startsWith('http://127.0.0.1:') || url.startsWith('file://');
    callback(isOwnApp && (permission === 'media' || permission === 'microphone'));
  });
  session.defaultSession.setPermissionCheckHandler((webContents, permission) => {
    const url = webContents?.getURL() || '';
    const isOwnApp = url.startsWith('http://127.0.0.1:') || url.startsWith('file://');
    return isOwnApp && (permission === 'media' || permission === 'microphone');
  });
});

app.whenReady().then(createWindow);
app.on('before-quit', () => { isQuitting = true; });
app.on('window-all-closed', () => { if (process.platform !== 'darwin') app.quit(); });
app.on('activate', () => { if (mainWindow) mainWindow.show(); else createWindow(); });
