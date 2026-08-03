// Local companion backend — runs on the patient's own device, port 8000,
// matching the `isLocal` / `location.hostname:8000` branch already present
// in the bundled app's own JS (unmodified). This is the piece that was
// actually missing: main.py's /session/stats and /session/aba-progress
// read a local filesystem path that only makes sense for a backend running
// ON THE DEVICE — nothing here sends therapy data anywhere. It stays on
// this machine, in this app's own userData directory, for this app's own
// use, consistent with the product's stated privacy design ("All
// therapeutic data lives ... on the user's device" — models.py's User
// docstring on the cloud backend).
//
// Implements exactly the two things the bundled page's own code already
// calls when running locally:
//   - GET  /session/new-code + WS /ws/broadcast/:code + WS /ws/monitor/:code
//     (ephemeral relay for live guardian monitoring on the same network —
//     ported faithfully from backend/main.py's in-memory relay, no storage)
//   - GET  /session/stats, GET /session/aba-progress
//     (real aggregate stats computed from locally-logged session metrics)

const http = require('http');
const fs = require('fs');
const path = require('path');
const { WebSocketServer } = require('ws');

function startLocalBackend(userDataDir) {
  const sessionsDir = path.join(userDataDir, 'sessions');
  fs.mkdirSync(sessionsDir, { recursive: true });
  const logFile = path.join(sessionsDir, 'session_log.jsonl');

  function appendMetric(metric) {
    fs.appendFileSync(logFile, JSON.stringify({ ...metric, loggedAt: Date.now() }) + '\n');
  }

  function readAllMetrics() {
    if (!fs.existsSync(logFile)) return [];
    return fs
      .readFileSync(logFile, 'utf8')
      .split('\n')
      .filter(Boolean)
      .map((line) => {
        try { return JSON.parse(line); } catch (e) { return null; }
      })
      .filter(Boolean);
  }

  // Real aggregate stats from real logged samples — no invented numbers.
  // Deliberately does NOT attempt to reproduce the original code's
  // "Cohen's d" / "VSA" / mastery-level metrics: that logic (science.py)
  // doesn't exist anywhere recoverable, and faking a statistical method
  // that was never actually implemented would be worse than reporting
  // only what's genuinely computed here.
  function computeStats() {
    const metrics = readAllMetrics();
    if (metrics.length === 0) {
      return { status: 'no_data', total_events: 0, sessions: [] };
    }
    const bySession = new Map();
    for (const m of metrics) {
      if (!bySession.has(m.sessionId)) bySession.set(m.sessionId, []);
      bySession.get(m.sessionId).push(m);
    }
    const sessions = [...bySession.entries()].map(([sessionId, rows]) => {
      const latencies = rows.map((r) => r.avgLat).filter((v) => typeof v === 'number');
      const hnrs = rows.map((r) => r.hnr).filter((v) => typeof v === 'number');
      const voicedCount = rows.filter((r) => r.isVoiced).length;
      const startedAt = Math.min(...rows.map((r) => r.loggedAt));
      const endedAt = Math.max(...rows.map((r) => r.loggedAt));
      const mean = (arr) => (arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : null);
      return {
        sessionId,
        started_at: new Date(startedAt).toISOString(),
        duration_seconds: Math.round((endedAt - startedAt) / 1000),
        sample_count: rows.length,
        voiced_ratio: +(voicedCount / rows.length).toFixed(3),
        mean_latency_ms: mean(latencies) !== null ? +mean(latencies).toFixed(1) : null,
        mean_hnr_db: mean(hnrs) !== null ? +mean(hnrs).toFixed(1) : null,
        total_corrections: Math.max(...rows.map((r) => r.correctionCount || 0)),
      };
    }).sort((a, b) => new Date(b.started_at) - new Date(a.started_at));

    return {
      status: 'ok',
      total_events: metrics.length,
      total_sessions: sessions.length,
      total_practice_seconds: sessions.reduce((a, s) => a + s.duration_seconds, 0),
      sessions,
    };
  }

  // --- Ephemeral relay: ported from backend/main.py, in-memory only ---
  const relay = new Map(); // code -> { broadcaster: ws|null, monitors: ws[] }

  const server = http.createServer((req, res) => {
    const url = new URL(req.url, 'http://localhost');
    res.setHeader('Access-Control-Allow-Origin', '*');

    if (url.pathname === '/session/new-code' && req.method === 'GET') {
      const code = require('crypto').randomBytes(3).toString('hex').toUpperCase();
      relay.set(code, { broadcaster: null, monitors: [] });
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ code }));
      return;
    }

    if (url.pathname === '/session/stats' && req.method === 'GET') {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(computeStats()));
      return;
    }

    if (url.pathname === '/session/aba-progress' && req.method === 'GET') {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({
        status: 'not_implemented',
        detail: 'ABA skill-mastery tracking needs per-attempt success/fail data this build does not yet capture — no skill-prompt UI exists. Not faked here.',
        skills: {},
      }));
      return;
    }

    res.writeHead(404);
    res.end();
  });

  const wss = new WebSocketServer({ server });
  wss.on('connection', (ws, req) => {
    const url = new URL(req.url, 'http://localhost');
    const m = url.pathname.match(/^\/ws\/(broadcast|monitor)\/([A-Z0-9]+)$/);
    if (!m) { ws.close(4004); return; }
    const [, kind, code] = m;
    if (!relay.has(code)) { ws.close(4004); return; }

    if (kind === 'broadcast') {
      relay.get(code).broadcaster = ws;
      ws.on('message', (data) => {
        const dead = [];
        for (const monitor of relay.get(code).monitors) {
          if (monitor.readyState === 1) monitor.send(data);
          else dead.push(monitor);
        }
        relay.get(code).monitors = relay.get(code).monitors.filter((m2) => !dead.includes(m2));
      });
      ws.on('close', () => {
        relay.get(code).broadcaster = null;
        for (const monitor of relay.get(code).monitors) {
          try { monitor.send('{"event":"session_ended"}'); } catch (e) {}
        }
      });
    } else {
      relay.get(code).monitors.push(ws);
      const ping = setInterval(() => {
        if (ws.readyState === 1) ws.send('{"event":"ping"}');
      }, 30000);
      ws.on('close', () => {
        clearInterval(ping);
        relay.get(code).monitors = relay.get(code).monitors.filter((m2) => m2 !== ws);
      });
    }
  });

  return new Promise((resolve, reject) => {
    server.on('error', reject);
    server.listen(8000, '127.0.0.1', () => resolve({ server, appendMetric, computeStats, logFile }));
  });
}

module.exports = { startLocalBackend };
