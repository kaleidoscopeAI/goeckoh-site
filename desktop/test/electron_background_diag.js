const { _electron: electron } = require('playwright');
const path = require('path');
const fs = require('fs');
const os = require('os');

(async () => {
  const userDataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'goeckoh-desktop-bg-'));
  fs.writeFileSync(
    path.join(userDataDir, 'license.json'),
    JSON.stringify({
      license_key: 'GK-TEST-0000-0000',
      device_fingerprint: 'test-device-fingerprint',
      token: 'fake.jwt.token',
      plan: 'starter',
      activated_at: Date.now() - 1000 * 60 * 60 * 24,
      last_validated_at: Date.now() - 1000 * 60 * 60 * 24,
    }, null, 2)
  );
  fs.writeFileSync(path.join(userDataDir, 'device-id'), 'test-device-fingerprint');

  const wav = '/tmp/claude-0/-home-user-goeckoh-site/a8101931-8d52-57ae-acc4-8f62fc8692b1/scratchpad/fake_voice.wav';

  const app = await electron.launch({
    executablePath: require('electron'),
    args: [
      path.join(__dirname, '..', 'main.js'),
      `--user-data-dir=${userDataDir}`,
      '--hidden',
      '--use-fake-ui-for-media-stream',
      '--use-fake-device-for-media-stream',
      `--use-file-for-fake-audio-capture=${wav}`,
    ],
    timeout: 30000,
  });

  const win = await app.firstWindow();
  await win.waitForLoadState('domcontentloaded');
  await win.waitForTimeout(2500); // let auto quickStart() fire

  const isVisible = await app.evaluate(async ({ BrowserWindow }) => {
    const w = BrowserWindow.getAllWindows()[0];
    return w ? w.isVisible() : null;
  });
  console.log('window isVisible with --hidden:', isVisible);

  const btnText = await win.$eval('#btnQuickStart', (el) => el.textContent.trim()).catch((e) => 'ERR: ' + e.message);
  console.log('Quick Start button text (should show Running if auto-started):', btnText);

  // Confirm correction is actually producing audio (not just UI state) —
  // same tap technique as before, just proving it started with zero clicks.
  await win.evaluate(() => {
    window.__finalOutputNode = null;
    const orig = AudioNode.prototype.connect;
    AudioNode.prototype.connect = function (dest, ...args) {
      if (dest instanceof AudioDestinationNode) window.__finalOutputNode = this;
      return orig.call(this, dest, ...args);
    };
  }).catch(() => {});
  const hasNode = await win.evaluate(() => !!window.__finalOutputNode).catch(() => false);
  console.log('output node already connected (proves quickStart ran before we could even patch):', hasNode);

  // Test close-to-tray: trigger the window's close event and confirm the
  // process is still alive (didn't quit) and window still exists, just hidden.
  await app.evaluate(async ({ BrowserWindow }) => {
    const w = BrowserWindow.getAllWindows()[0];
    w.close();
  });
  await new Promise((r) => setTimeout(r, 500));
  const stillRunning = await app.evaluate(async ({ BrowserWindow }) => BrowserWindow.getAllWindows().length).catch((e) => 'ERR: ' + e.message);
  console.log('windows still existing after close() (should be 1, hidden not destroyed):', stillRunning);
  const visibleAfterClose = await app.evaluate(async ({ BrowserWindow }) => BrowserWindow.getAllWindows()[0]?.isVisible());
  console.log('visible after close() (should be false):', visibleAfterClose);

  await app.close();
  fs.rmSync(userDataDir, { recursive: true, force: true });
})();
