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

  console.log('Clicking Quick Start...');
  await win.click('#btnQuickStart');
  await win.waitForTimeout(1000);

  const hasOutputNode = await win.evaluate(() => !!window.__finalOutputNode);
  console.log('captured final output node:', hasOutputNode);

  if (!hasOutputNode) {
    console.log('=== CONSOLE ===');
    console.log(consoleMsgs.join('\n') || '(none)');
    await win.screenshot({ path: path.join(__dirname, 'screenshot-audio-fail.png') });
    await app.close();
    return;
  }

  const envelope = await win.evaluate(async () => {
    return await new Promise((resolve) => {
      const ctx = window.__finalOutputNode.context;
      const results = [];
      const winSize = Math.round(ctx.sampleRate * 0.1);
      const sp = ctx.createScriptProcessor(2048, 1, 1);
      let buf = [];
      let sawNaN = false, maxAbs = 0, clippedSamples = 0, totalSamples = 0;
      sp.onaudioprocess = (e) => {
        const inData = e.inputBuffer.getChannelData(0);
        for (let i = 0; i < inData.length; i++) {
          const v = inData[i];
          totalSamples++;
          if (Number.isNaN(v) || !Number.isFinite(v)) sawNaN = true;
          const av = Math.abs(v);
          if (av > maxAbs) maxAbs = av;
          if (av >= 0.999) clippedSamples++;
          buf.push(v);
        }
        while (buf.length >= winSize) {
          const win_ = buf.slice(0, winSize);
          buf = buf.slice(winSize);
          let sumSq = 0, peak = 0;
          for (const s of win_) { sumSq += s * s; if (Math.abs(s) > peak) peak = Math.abs(s); }
          results.push({ rms: +Math.sqrt(sumSq / win_.length).toFixed(5), peak: +peak.toFixed(5) });
        }
      };
      const sink = ctx.createGain();
      sink.gain.value = 0;
      window.__finalOutputNode.connect(sp);
      sp.connect(sink);
      sink.connect(ctx.destination);
      setTimeout(() => {
        try { window.__finalOutputNode.disconnect(sp); } catch (e) {}
        try { sp.disconnect(); } catch (e) {}
        resolve({ windows: results, sawNaN, maxAbs, clippedSamples, totalSamples, sampleRate: ctx.sampleRate });
      }, 8000);
    });
  });

  console.log('=== CONSOLE ===');
  console.log(consoleMsgs.join('\n') || '(none)');
  console.log('=== AUDIO SUMMARY ===');
  console.log('sawNaN:', envelope.sawNaN);
  console.log('maxAbs (true peak):', envelope.maxAbs);
  console.log('clippedSamples:', envelope.clippedSamples, '/', envelope.totalSamples,
    '(' + (100 * envelope.clippedSamples / envelope.totalSamples).toFixed(1) + '%)');
  console.log('window count:', envelope.windows.length);
  console.log('first 5:', JSON.stringify(envelope.windows.slice(0, 5)));
  console.log('last 5:', JSON.stringify(envelope.windows.slice(-5)));

  await win.screenshot({ path: path.join(__dirname, 'screenshot-audio-run.png') });
  await app.close();
  fs.rmSync(userDataDir, { recursive: true, force: true });
})();
