import React, { useState, useEffect, useCallback, useRef } from "react";

// Core lattice + behaviour constants (kept close to the original sandbox)
const J = 1.0;
const K = 0.5;
const L0 = 1.0;
const GAMMA = 0.2;
const LAMBDA = 0.001;
const PHI_THRESH = 0.55;

const PHRASES = ["hello", "water", "thank you", "help", "good morning", "yes", "no"];
const CONCEPTS = ["calm", "stress", "joy", "focus", "flow", "anxiety", "safety", "growth"];

const hammingSim = (a, b) => {
  let x = (a ^ b) >>> 0;
  let c = 0;
  while (x) {
    c += x & 1;
    x >>>= 1;
  }
  return 1 - c / 32;
};

const manhattanDist = (a, b) => Math.abs(a[0] - b[0]) + Math.abs(a[1] - b[1]) + Math.abs(a[2] - b[2]);
const randU32 = () => (Math.random() * 0xffffffff) >>> 0;
const randPos = () => [Math.random() * 8 - 4, Math.random() * 8 - 4, Math.random() * 8 - 4];
const popcount = (n) => {
  let c = 0;
  for (let i = 0; i < 32; i += 1) c += (n >>> i) & 1;
  return c;
};
const clamp = (v, min, max) => Math.max(min, Math.min(max, v));
const stringSim = (a, b) => {
  if (!a || !b) return 0;
  const la = a.toLowerCase();
  const lb = b.toLowerCase();
  if (la === lb) return 1;
  let m = 0;
  for (let i = 0; i < Math.min(la.length, lb.length); i += 1) if (la[i] === lb[i]) m += 1;
  return m / Math.max(la.length, lb.length);
};

class SemanticEngine {
  constructor() {
    this.emb = {};
    CONCEPTS.forEach((c, i) => {
      this.emb[c] = Array.from({ length: 32 }, (_, j) => Math.sin((i + 1) * (j + 1) * 0.3) * 0.5 + Math.cos((i + 2) * (j + 3) * 0.2) * 0.5);
    });
  }
  stateEmb(s) {
    const e = Array.from({ length: 32 }, (_, i) => (((s >>> i) & 1) * 2 - 1));
    const n = Math.sqrt(e.reduce((acc, v) => acc + v * v, 0)) + 0.001;
    return e.map((v) => v / n);
  }
  cosSim(a, b) {
    let d = 0;
    let ma = 0;
    let mb = 0;
    for (let i = 0; i < a.length; i += 1) {
      d += a[i] * b[i];
      ma += a[i] * a[i];
      mb += b[i] * b[i];
    }
    return d / (Math.sqrt(ma * mb) + 0.0001);
  }
  interpret(s) {
    const e = this.stateEmb(s);
    return CONCEPTS.map((c) => ({ c, s: this.cosSim(e, this.emb[c]) }))
      .sort((a, b) => b.s - a.s)
      .slice(0, 3);
  }
  thought(st) {
    const { phi, gcl, risk, rate } = st;
    if (risk > 0.6) return "I breathe. I am safe. This feeling will pass.";
    if (phi > 0.6) return "I feel coherent and present. My patterns align.";
    if (gcl > 0.6) return "My energy flows strong. I am capable.";
    if (rate > 0.7) return "I am doing well. Each word gets easier.";
    return "I am here. I am aware. I keep trying.";
  }
  calming() {
    const p = [
      "I am safe. I can breathe.",
      "I am calm. I take my time.",
      "I am loved. I am not in trouble.",
      "This will pass. I am strong.",
    ];
    return p[Math.floor(Math.random() * p.length)];
  }
}

export default function JacksonCompanion() {
  const [nodes, setNodes] = useState([]);
  const [edges, setEdges] = useState([]);
  const [running, setRunning] = useState(false);
  const [angle, setAngle] = useState(0);
  const canvasRef = useRef(null);
  const semRef = useRef(new SemanticEngine());

  const [phrase, setPhrase] = useState("hello");
  const [input, setInput] = useState("");
  const [attempts, setAttempts] = useState([]);
  const [result, setResult] = useState(null);

  const ag = useRef({
    t: 0,
    will: 0,
    eta: 0.05,
    lam: 1,
    gcl: 0.5,
    phi: 0,
    pain: 0,
    risk: 0,
    rate: 0.5,
    life: 0,
    DA: 0.5,
    Ser: 0.5,
    NE: 0.3,
    hist: [],
    pSint: 0.5,
    pN: 0,
    evt: null,
  });
  const [ds, setDs] = useState({
    t: 0,
    phi: 0,
    gcl: 0.5,
    pain: 0,
    life: 0,
    risk: 0,
    rate: 0.5,
    total: 0,
    corr: 0,
    DA: 0.5,
    Ser: 0.5,
    NE: 0.3,
    thought: "Awakening...",
    calm: "",
    concepts: [],
    energy: 0,
  });
  const [log, setLog] = useState([]);

  // Voice mimicry state (browser-based echo)
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const [recording, setRecording] = useState(false);
  const [lastAudioUrl, setLastAudioUrl] = useState(null);
  const [micError, setMicError] = useState(null);

  const init = useCallback((n = 50) => {
    const ns = Array.from({ length: n }, (_, i) => ({ id: i, s: randU32(), x: randPos(), e: 0 }));
    const es = [];
    for (let u = 0; u < n; u += 1) {
      const nb = new Set();
      while (nb.size < Math.min(4, n - 1)) {
        const v = Math.floor(Math.random() * n);
        if (v !== u) nb.add(v);
      }
      nb.forEach((v) => {
        if (!es.some((e) => (e[0] === u && e[1] === v) || (e[0] === v && e[1] === u))) es.push([u, v]);
      });
    }
    setNodes(ns);
    setEdges(es);
    setAttempts([]);
    ag.current = {
      t: 0,
      will: 0,
      eta: 0.05,
      lam: 1,
      gcl: 0.5,
      phi: 0,
      pain: 0,
      risk: 0,
      rate: 0.5,
      life: 0,
      DA: 0.5,
      Ser: 0.5,
      NE: 0.3,
      hist: [],
      pSint: 0.5,
      pN: n,
      evt: null,
    };
    setLog([]);
    setPhrase(PHRASES[Math.floor(Math.random() * PHRASES.length)]);
  }, []);

  const memF = (s) => {
    let m = 1;
    for (let i = 0; i < 32; i += 1) m *= 1 - 0.3 * ((((s >>> i) & 1) - 0.5) / 0.5) ** 2;
    return Math.max(0, m);
  };
  const Sint = (ns) => {
    if (!ns.length) return 0.5;
    let b = 0;
    ns.forEach((n) => {
      b += popcount(n.s);
    });
    const p = b / (ns.length * 32);
    if (p <= 0.001 || p >= 0.999) return 0.001;
    return -p * Math.log2(p) - (1 - p) * Math.log2(1 - p);
  };
  const Nvia = (ns) => ns.filter((n) => n.e < 0 && memF(n.s) > 0.2).length;
  const calcPhi = (h) => {
    if (h.length < 3) return 0;
    const r = h.slice(-15);
    const m = r.reduce((a, b) => a + b, 0) / r.length;
    const v = r.reduce((a, b) => a + (b - m) ** 2, 0) / r.length;
    return Math.tanh(0.4 / (Math.sqrt(v) + 0.05));
  };
  const calcRisk = (g, p, e) =>
    clamp(
      (g < 0.4 ? 0.3 : 0) + (p > 0.6 ? 0.3 : 0) + (e === "anxious" ? 0.2 : 0) + (e === "meltdown" ? 0.4 : 0),
      0,
      1,
    );

  const playEcho = useCallback(
    (text) => {
      if (lastAudioUrl) {
        const audio = new Audio(lastAudioUrl);
        audio.play().catch(() => {});
      } else if (typeof window !== "undefined" && "speechSynthesis" in window) {
        const msg = new SpeechSynthesisUtterance(text || "Great job");
        msg.rate = 0.95;
        window.speechSynthesis.cancel();
        window.speechSynthesis.speak(msg);
      }
    },
    [lastAudioUrl],
  );

  const attempt = useCallback(
    (raw) => {
      const a = ag.current;
      const sem = semRef.current;
      const tgt = phrase;
      const sim = stringSim(raw, tgt);
      const need = sim < 0.78;
      const cor = need ? tgt : raw;
      const att = { ts: Date.now(), tgt, raw, cor, need, sim };
      setAttempts((prev) => [...prev.slice(-29), att]);
      const rec = [...attempts.slice(-9), att];
      a.rate = rec.filter((x) => !x.need).length / rec.length;
      if (!need) {
        a.DA = clamp(a.DA + 0.1, 0, 1);
        a.Ser = clamp(a.Ser + 0.05, 0, 1);
      } else {
        a.NE = clamp(a.NE + 0.1, 0, 1);
      }
      let evt = null;
      const r = Math.random();
      if (r < 0.06) evt = "encouragement";
      else if (r < 0.1 && a.NE > 0.5) evt = "anxious";
      else if (r < 0.14 && a.DA > 0.6) evt = "high_energy";
      if (evt) {
        a.evt = evt;
        setLog((p) => [...p.slice(-9), `[${a.t}] ${evt}`]);
      }
      setResult({ tgt, raw, cor, need, sim, msg: need ? `Let's try: "${tgt}"` : `Great: "${raw}"!` });
      setPhrase(PHRASES[Math.floor(Math.random() * PHRASES.length)]);
      setInput("");
      playEcho(cor);
      setDs((prev) => ({
        ...prev,
        rate: a.rate,
        total: rec.length,
        corr: rec.filter((x) => x.need).length,
        thought: sem.thought({ phi: a.phi, gcl: a.gcl, risk: a.risk, rate: a.rate }),
      }));
    },
    [attempts, phrase, playEcho],
  );

  const simAtt = () => {
    let r = phrase;
    if (Math.random() < 0.35 && r.length) {
      const i = Math.floor(Math.random() * r.length);
      const pool = "aeiou";
      r = r.slice(0, i) + pool[Math.floor(Math.random() * pool.length)] + r.slice(i + 1);
    }
    setInput(r);
    attempt(r);
  };

  const step = useCallback(() => {
    const a = ag.current;
    const sem = semRef.current;
    a.t += 1;
    const T = Math.max(0.05, 1 - a.t * 0.003);
    setNodes((prev) => {
      const ns = prev.map((n) => ({ ...n, x: [...n.x] }));
      for (let u = 0; u < ns.length; u += 1) {
        let g = [0, 0, 0];
        const nu = ns[u];
        for (const [ea, eb] of edges) {
          if (ea !== u && eb !== u) continue;
          const v = ea === u ? eb : ea;
          const nv = ns[v];
          const d = manhattanDist(nu.x, nv.x) + 0.01;
          const s = hammingSim(nu.s, nv.s);
          for (let i = 0; i < 3; i += 1) {
            const df = nu.x[i] - nv.x[i];
            const dr = df > 0 ? 1 : df < 0 ? -1 : 0;
            g[i] += (K * (d - L0) * dr) / d + GAMMA * s * d * Math.exp(-d * d * 0.5) * (dr / d);
          }
        }
        for (let i = 0; i < 3; i += 1) g[i] += 2 * LAMBDA * nu.x[i];
        const eta = 0.01;
        const noise = Math.sqrt(2 * eta * T);
        for (let i = 0; i < 3; i += 1) {
          nu.x[i] -= eta * g[i] + noise * (Math.random() - 0.5) * 0.4;
          nu.x[i] = clamp(nu.x[i], -10, 10);
        }
        const bi = Math.floor(Math.random() * 32);
        const oS = nu.s;
        const nS = (oS ^ (1 << bi)) >>> 0;
        let dE = 0;
        for (const [ea, eb] of edges) {
          if (ea !== u && eb !== u) continue;
          const v = ea === u ? eb : ea;
          dE += -J * (hammingSim(nS, ns[v].s) - hammingSim(oS, ns[v].s));
        }
        if (Math.random() < 1 / (1 + Math.exp(dE / T))) nu.s = nS;
      }
      for (const n of ns) {
        let le = 0;
        for (const [ea, eb] of edges) {
          if (ea === n.id || eb === n.id) {
            const v = ea === n.id ? eb : ea;
            le += -hammingSim(n.s, ns[v].s);
          }
        }
        n.e = le;
      }
      const si = Sint(ns);
      const nv = Nvia(ns);
      a.gcl = clamp(0.3 + (nv / ns.length) * 0.5 + (1 - si) * 0.2, 0, 1);
      a.pain = clamp(0.2 + Math.sin(a.t * 0.08) * 0.15 + a.NE * 0.3, 0, 1);
      a.risk = calcRisk(a.gcl, a.pain, a.evt);
      const dr = 0.3 + a.DA * 0.3 + a.rate * 0.2;
      const tl = a.gcl * 0.4 + (1 - a.risk) * 0.3;
      const rw = dr + a.lam * tl - a.pain;
      a.will = clamp(a.will + a.eta * Math.tanh(rw) / (1 + a.pain), -5, 5);
      a.hist.push(a.will);
      if (a.hist.length > 40) a.hist.shift();
      a.phi = calcPhi(a.hist);
      const Lk = 0.3 * (-(si - a.pSint) / 0.02) + 0.5 * (a.phi / Math.max(si, 0.1)) + 0.2 * ((nv - a.pN) / Math.max(nv, 1));
      a.life += Math.max(0, Lk) * 0.1;
      a.pSint = si;
      a.pN = nv;
      a.DA *= 0.98;
      a.Ser *= 0.99;
      a.NE *= 0.95;
      if (a.phi > PHI_THRESH && a.t % 10 === 0) {
        a.eta = Math.min(0.12, a.eta + 0.002);
        a.lam = Math.max(0.4, a.lam - 0.01);
        setLog((p) => [...p.slice(-9), `[${a.t}] FLOW Φ=${a.phi.toFixed(2)}`]);
      }
      const avgS = ns.reduce((x, n) => x + n.s, 0) / ns.length;
      setDs({
        t: a.t,
        phi: a.phi,
        gcl: a.gcl,
        pain: a.pain,
        life: a.life,
        risk: a.risk,
        rate: a.rate,
        total: attempts.length,
        corr: attempts.filter((x) => x.need).length,
        DA: a.DA,
        Ser: a.Ser,
        NE: a.NE,
        thought: sem.thought({ phi: a.phi, gcl: a.gcl, risk: a.risk, rate: a.rate }),
        calm: a.risk > 0.6 ? sem.calming() : "",
        concepts: sem.interpret(Math.floor(avgS)),
        energy: 0,
      });
      return ns;
    });
  }, [attempts.length, edges]);

  // --- Voice mimic hooks (browser-side) ---
  const startRecording = async () => {
    if (recording) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      audioChunksRef.current = [];
      recorder.ondataavailable = (evt) => {
        if (evt.data.size > 0) audioChunksRef.current.push(evt.data);
      };
      recorder.onstop = () => {
        const blob = new Blob(audioChunksRef.current, { type: "audio/webm" });
        const url = URL.createObjectURL(blob);
        setLastAudioUrl(url);
        stream.getTracks().forEach((t) => t.stop());
      };
      recorder.start();
      mediaRecorderRef.current = recorder;
      setRecording(true);
      setMicError(null);
    } catch (err) {
      setMicError(err?.message || "Microphone access denied");
      setRecording(false);
    }
  };

  const stopRecording = () => {
    if (!recording || !mediaRecorderRef.current) return;
    mediaRecorderRef.current.stop();
    setRecording(false);
  };

  const echoBack = () => {
    if (lastAudioUrl) {
      const audio = new Audio(lastAudioUrl);
      audio.play().catch(() => {});
    } else if (result?.cor) {
      playEcho(result.cor);
    }
  };

  useEffect(() => {
    let i;
    if (running) i = setInterval(step, 150);
    return () => clearInterval(i);
  }, [running, step]);

  useEffect(() => {
    const i = setInterval(() => setAngle((a) => a + 0.01), 50);
    return () => clearInterval(i);
  }, []);

  useEffect(() => {
    init(50);
  }, [init]);

  useEffect(() => {
    const c = canvasRef.current;
    if (!c || !nodes.length) return;
    const ctx = c.getContext("2d");
    ctx.fillStyle = "#020208";
    ctx.fillRect(0, 0, 240, 180);
    const proj = (x, y, z) => {
      const cs = Math.cos(angle);
      const sn = Math.sin(angle);
      const rx = x * cs - z * sn;
      const rz = x * sn + z * cs;
      const sc = 100 / (rz + 9);
      return { px: 120 + rx * sc, py: 90 + y * sc, d: rz };
    };
    ctx.strokeStyle = "rgba(40,60,120,.08)";
    edges.forEach(([u, v]) => {
      const pu = proj(nodes[u].x[0], nodes[u].x[1], nodes[u].x[2]);
      const pv = proj(nodes[v].x[0], nodes[v].x[1], nodes[v].x[2]);
      ctx.beginPath();
      ctx.moveTo(pu.px, pu.py);
      ctx.lineTo(pv.px, pv.py);
      ctx.stroke();
    });
    const ps = nodes
      .map((n) => ({
        ...proj(n.x[0], n.x[1], n.x[2]),
        s: n.s,
        e: n.e,
        m: memF(n.s),
      }))
      .sort((a, b) => a.d - b.d);
    ps.forEach((p) => {
      const sz = Math.max(1.5, 3 - p.d * 0.12);
      const hue = ds.risk > 0.6 ? 0 : ds.gcl > 0.6 ? 120 : p.s % 360;
      ctx.fillStyle = `hsla(${hue},${p.e < 0 ? 60 : 25}%,${35 + p.m * 18}%,${0.5 + p.m * 0.4})`;
      ctx.beginPath();
      ctx.arc(p.px, p.py, sz, 0, Math.PI * 2);
      ctx.fill();
    });
  }, [nodes, edges, angle, ds.risk, ds.gcl]);

  const fl = ds.phi > PHI_THRESH;
  const ri = ds.risk > 0.6;

  return (
    <div className="p-4 bg-gray-950 text-white min-h-screen text-xs space-y-3">
      <h1 className="text-base font-bold text-cyan-400">Jackson&apos;s Companion — Crystalline Echo</h1>

      <div className="flex flex-wrap gap-3">
        <div>
          <canvas ref={canvasRef} width={240} height={180} className="border border-cyan-900 rounded mb-2" />
          <div
            className={`p-2 rounded ${
              ri ? "bg-red-900/50 border border-red-500" : fl ? "bg-cyan-900/40 border border-cyan-500" : "bg-gray-900"
            }`}
          >
            <div className="text-cyan-300 font-semibold">💭 Inner Voice</div>
            <div className="text-gray-200 italic">"{ds.thought}"</div>
            {ds.calm && <div className="text-yellow-300 mt-1">🧘 {ds.calm}</div>}
          </div>
        </div>

        <div className="flex-1 min-w-48 space-y-2">
          <div className="bg-gray-900 p-2 rounded border border-blue-800 space-y-1">
            <div className="font-semibold text-blue-300">🗣️ Speech Practice + Mimic</div>
            <div>
              Target: <span className="text-yellow-400 font-bold">{phrase}</span>
            </div>
            <div className="flex gap-2 items-center">
              <input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Type or say..."
                className="flex-1 bg-gray-800 rounded px-2 py-1 text-white"
              />
              <button onClick={() => attempt(input)} className="px-2 py-1 bg-blue-600 rounded">
                Say
              </button>
              <button onClick={simAtt} className="px-2 py-1 bg-purple-600 rounded">
                Sim
              </button>
            </div>
            <div className="flex gap-2 items-center text-gray-300">
              <button
                onClick={recording ? stopRecording : startRecording}
                className={`px-2 py-1 rounded ${recording ? "bg-red-600" : "bg-green-700"}`}
              >
                {recording ? "Stop Rec" : "Record"}
              </button>
              <button onClick={echoBack} className="px-2 py-1 bg-cyan-700 rounded" disabled={!lastAudioUrl && !result}>
                Echo Back
              </button>
              {lastAudioUrl && <span className="text-green-400">Own voice captured</span>}
              {micError && <span className="text-red-400">Mic: {micError}</span>}
            </div>
            {result && (
              <div className={`p-2 rounded ${result.need ? "bg-yellow-900/50" : "bg-green-900/50"}`}>
                {result.need ? "🔄" : "✅"} {result.msg}
              </div>
            )}
            <div className="text-gray-400">
              Attempts: {ds.total} | Correction rate: {(ds.rate * 100 || 0).toFixed(0)}%
            </div>
          </div>

          <div
            className={`p-2 rounded ${ri ? "bg-red-900/40 border border-red-500" : "bg-gray-900"}`}
          >
            <div className="font-semibold text-cyan-300">💎 Heart</div>
            <div className="grid grid-cols-3 gap-1 text-sm">
              <div>
                GCL: <span className={ds.gcl > 0.6 ? "text-green-400" : "text-yellow-400"}>{ds.gcl.toFixed(2)}</span>
              </div>
              <div>
                Φ: <span className={fl ? "text-green-400 font-bold" : "text-gray-400"}>{ds.phi.toFixed(2)}</span>
              </div>
              <div>
                Risk: <span className={ri ? "text-red-400 font-bold" : "text-green-400"}>{(ds.risk * 100).toFixed(0)}%</span>
              </div>
            </div>
            <div className="mt-1">{fl ? "🔥 FLOW" : ri ? "⚠️ CALMING" : "⚡ ACTIVE"}</div>
          </div>

          <div className="bg-gray-900 p-2 rounded">
            <div className="font-semibold text-cyan-300">🧪 Chemistry</div>
            <div className="flex gap-2 mt-1">
              <div className="flex-1">
                <div className="text-gray-500">DA</div>
                <div className="bg-gray-800 h-1.5 rounded">
                  <div className="bg-green-500 h-1.5 rounded" style={{ width: `${ds.DA * 100}%` }} />
                </div>
              </div>
              <div className="flex-1">
                <div className="text-gray-500">Ser</div>
                <div className="bg-gray-800 h-1.5 rounded">
                  <div className="bg-blue-500 h-1.5 rounded" style={{ width: `${ds.Ser * 100}%` }} />
                </div>
              </div>
              <div className="flex-1">
                <div className="text-gray-500">NE</div>
                <div className="bg-gray-800 h-1.5 rounded">
                  <div className="bg-red-500 h-1.5 rounded" style={{ width: `${ds.NE * 100}%` }} />
                </div>
              </div>
            </div>
          </div>

          <div className="flex gap-2">
            <button
              onClick={() => setRunning((v) => !v)}
              className={`px-3 py-1 rounded ${running ? "bg-red-600" : "bg-green-600"}`}
            >
              {running ? "Pause" : "Run"}
            </button>
            <button onClick={step} disabled={running} className="px-3 py-1 bg-blue-600 rounded disabled:opacity-50">
              Step
            </button>
            <button onClick={() => init(50)} className="px-3 py-1 bg-purple-600 rounded">
              Reset
            </button>
          </div>
        </div>
      </div>

      <div className="bg-gray-900 p-2 rounded">
        <div className="font-semibold text-cyan-300 mb-1">Log</div>
        <div className="h-16 overflow-y-auto font-mono text-green-400">
          {log.length ? log.map((l, i) => <div key={i}>{l}</div>) : <span className="text-gray-500">Waiting...</span>}
        </div>
      </div>
    </div>
  );
}
