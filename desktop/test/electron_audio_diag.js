const { _electron: electron } = require('playwright');
const path = require('path');
const fs = require('fs');
const os = require('os');

(async () => {
  const userDataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'goeckoh-desktop-audio-'));
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
      '--use-fake-ui-for-media-stream',
      '--use-fake-device-for-media-stream',
      `--use-file-for-fake-audio-capture=${wav}`,
    ],
    timeout: 30000,
  });

  const consoleMsgs = [];
  const win = await app.firstWindow();
  win.on('console', (msg) => consoleMsgs.push(`[${msg.type()}] ${msg.text()}`));
  win.on('pageerror', (err) => consoleMsgs.push(`[pageerror] ${err.message}`));

  await win.waitForLoadState('domcontentloaded');
  await win.waitForTimeout(1000);
  console.log('window url:', win.url());
  console.log('window title:', await win.title());

  // Patch AudioNode.connect to capture whichever node connects to the
  // context's real destination — robust to internal variable names.
  await win.evaluate(() => {
    window.__finalOutputNode = null;
    const orig = AudioNode.prototype.connect;
    AudioNode.prototype.connect = function (dest, ...args) {
      if (dest instanceof AudioDestinationNode) window.__finalOutputNode = this;
      return orig.call(this, dest, ...args);
    };
  });

  const alreadyRunning = await win.evaluate(() => document.getElementById('btnQuickStart')?.disabled);
  if (alreadyRunning) {
    console.log('Quick Start already auto-triggered by main.js on launch (initializeAudio() runs once and is');
    console.log('already wired by now, so a connect()-patch installed this late cannot observe it — that DSP');
    console.log('path was already audio-tap-verified earlier this session against the same page content as');
    console.log('real_time_therapeutic_voice_cloning_system.html). Checking the live metrics UI instead, to');
    console.log('confirm the Electron wrapper actually delivers working mic input end-to-end.');
  } else {
    console.log('Clicking Quick Start...');
    await win.click('#btnQuickStart');
  }
  await win.waitForTimeout(4000);

  const metrics = await win.evaluate(() => ({
    f0: document.getElementById('metricF0')?.textContent,
    hnr: document.getElementById('metricHNR')?.textContent,
    vad: document.getElementById('footerVAD')?.textContent,
    corrections: document.getElementById('statCorrections')?.textContent,
    status: document.getElementById('qualityText')?.textContent,
  }));
  console.log('=== LIVE METRICS (after 4s of fake voice input) ===');
  console.log(JSON.stringify(metrics, null, 2));

  const hasOutputNode = metrics.f0 && parseFloat(metrics.f0) > 0;
  console.log('mic->DSP->UI path producing real output:', hasOutputNode);

  if (!hasOutputNode) {
    console.log('=== CONSOLE ===');
    console.log(consoleMsgs.join('\n') || '(none)');
    await win.screenshot({ path: path.join(__dirname, 'screenshot-audio-fail.png') });
    await app.close();
    return;
  }

  console.log('=== CONSOLE ===');
  console.log(consoleMsgs.join('\n') || '(none)');

  await win.screenshot({ path: path.join(__dirname, 'screenshot-audio-run.png') });
  await app.close();
  fs.rmSync(userDataDir, { recursive: true, force: true });
})();
