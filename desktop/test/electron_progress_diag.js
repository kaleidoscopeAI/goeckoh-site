const { _electron: electron } = require('playwright');
const path = require('path');
const fs = require('fs');
const os = require('os');
const http = require('http');

function get(url) {
  return new Promise((resolve, reject) => {
    http.get(url, (res) => {
      let raw = '';
      res.on('data', (c) => (raw += c));
      res.on('end', () => {
        try { resolve(JSON.parse(raw)); } catch (e) { reject(new Error('bad json: ' + raw)); }
      });
    }).on('error', reject);
  });
}

(async () => {
  const userDataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'goeckoh-desktop-progress-'));
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

  const win = await app.firstWindow();
  await win.waitForLoadState('domcontentloaded');
  console.log('Auto-starting, letting it run 6s to accumulate real metric samples...');
  await win.waitForTimeout(6000);

  console.log('\n=== GET /session/stats (before stop) ===');
  const stats = await get('http://127.0.0.1:8000/session/stats');
  console.log(JSON.stringify(stats, null, 2));

  console.log('\n=== GET /session/aba-progress (should be honest not_implemented) ===');
  const aba = await get('http://127.0.0.1:8000/session/aba-progress');
  console.log(JSON.stringify(aba, null, 2));

  console.log('\n=== GET /session/new-code + relay smoke test ===');
  const codeResp = await get('http://127.0.0.1:8000/session/new-code');
  console.log('code:', codeResp.code);

  const WebSocket = require('ws');
  const relayResult = await new Promise((resolve) => {
    const monitor = new WebSocket(`ws://127.0.0.1:8000/ws/monitor/${codeResp.code}`);
    const broadcaster = new WebSocket(`ws://127.0.0.1:8000/ws/broadcast/${codeResp.code}`);
    let received = null;
    monitor.on('message', (data) => { received = data.toString(); });
    broadcaster.on('open', () => {
      setTimeout(() => broadcaster.send(JSON.stringify({ f0: 123, f1: 500, f2: 1500 })), 300);
    });
    setTimeout(() => {
      monitor.close(); broadcaster.close();
      resolve(received);
    }, 1200);
  });
  console.log('monitor received from broadcaster via relay:', relayResult);

  const logFileContents = fs.readFileSync(path.join(userDataDir, 'sessions', 'session_log.jsonl'), 'utf8');
  const lines = logFileContents.trim().split('\n');
  console.log('\n=== raw local log file: ' + lines.length + ' lines, sample: ===');
  console.log(lines[0]);
  console.log(lines[lines.length - 1]);

  await app.close();
  fs.rmSync(userDataDir, { recursive: true, force: true });
})();
