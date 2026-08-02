const { _electron: electron } = require('playwright');
const path = require('path');
const fs = require('fs');
const os = require('os');

async function run(label, seedLicense) {
  const userDataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'goeckoh-desktop-test-'));
  if (seedLicense) {
    fs.writeFileSync(
      path.join(userDataDir, 'license.json'),
      JSON.stringify({
        license_key: 'GK-TEST-0000-0000',
        device_fingerprint: 'test-device-fingerprint',
        token: 'fake.jwt.token',
        plan: 'starter',
        activated_at: Date.now() - 1000 * 60 * 60 * 24, // "activated" a day ago
        last_validated_at: Date.now() - 1000 * 60 * 60 * 24,
      }, null, 2)
    );
    fs.writeFileSync(path.join(userDataDir, 'device-id'), 'test-device-fingerprint');
  }

  console.log(`\n=== ${label} (userData: ${userDataDir}) ===`);

  const app = await electron.launch({
    executablePath: require('electron'),
    args: [
      path.join(__dirname, '..', 'main.js'),
      `--user-data-dir=${userDataDir}`,
      '--use-fake-ui-for-media-stream',
      '--use-fake-device-for-media-stream',
    ],
    timeout: 30000,
  });

  const consoleMsgs = [];
  app.process().stdout.on('data', (d) => consoleMsgs.push('[main-stdout] ' + d.toString().trim()));
  app.process().stderr.on('data', (d) => consoleMsgs.push('[main-stderr] ' + d.toString().trim()));
  const win = await app.firstWindow();
  win.on('console', (msg) => consoleMsgs.push(`[${msg.type()}] ${msg.text()}`));
  win.on('pageerror', (err) => consoleMsgs.push(`[pageerror] ${err.message}`));

  await win.waitForLoadState('domcontentloaded');
  await win.waitForTimeout(2000);

  const url = win.url();
  const title = await win.title().catch(() => '(no title)');
  console.log('window url:', url);
  console.log('window title:', title);
  console.log('console/errors:', consoleMsgs.join('\n') || '(none)');

  await win.screenshot({ path: path.join(__dirname, `screenshot-${label.replace(/\s+/g, '_')}.png`) });

  await app.close();
  fs.rmSync(userDataDir, { recursive: true, force: true });
}

(async () => {
  await run('clean_no_license', false);
  await run('seeded_cached_license', true);
})();
