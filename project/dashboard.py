"""Rich, responsive caregiver dashboard for the autism speech companion."""

from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import librosa

from flask import Flask, jsonify, render_template_string, request

from .advanced_voice_mimic import VoiceProfile, TTSEngine
from .settings_store import SettingsStore
import tempfile
import os
import soundfile as sf

from .config import CompanionConfig, CONFIG
from .calming_strategies import STRATEGIES
from .policy import GuardianPolicy


TEMPLATE = """<!doctype html>
<html lang="en" class="h-full">
  <head>
    <meta charset="utf-8" />
    <title>Speech Companion • Care Dashboard</title>
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
      :root {
        color-scheme: dark;
      }
      .glass {
        background: radial-gradient(circle at top left, rgba(56, 189, 248, 0.22), transparent 55%),
                    radial-gradient(circle at bottom right, rgba(99, 102, 241, 0.18), transparent 55%);
      }
    </style>
  </head>
  <body class="h-full bg-slate-950 text-slate-50">
    <div class="min-h-screen px-4 py-6 sm:px-8 lg:px-12 glass">
      <!-- Header -->
      <header class="flex flex-col gap-4 md:flex-row md:items-center md:justify-between mb-8">
        <div>
          <p class="text-xs uppercase tracking-[0.3em] text-sky-400/70">Autism Speech Companion</p>
          <h1 class="text-3xl sm:text-4xl font-semibold tracking-tight mt-1">Care Dashboard</h1>
          <p class="text-sm text-slate-300/80 mt-3 max-w-xl">
            Live insight into practice sessions, correction patterns, and calming support —
            designed so caregivers, therapists, and researchers can see progress at a glance.
          </p>
        </div>
        <div class="flex flex-col items-start md:items-end gap-2">
          <div class="inline-flex items-center gap-2 rounded-full border border-emerald-400/40 bg-emerald-500/10 px-3 py-1">
            <span class="h-2 w-2 rounded-full bg-emerald-400 animate-pulse"></span>
            <span class="text-xs font-medium tracking-wide text-emerald-100">Live companion ready</span>
          </div>
          <div class="text-right text-xs text-slate-300/80">
            <div class="font-medium text-sm">{{ child_name }}</div>
            <div>Total attempts: <span class="font-semibold">{{ total_attempts }}</span></div>
            <div>Overall correction rate:
              <span class="font-semibold">
                {% if total_attempts %}
                  {{ '%.0f%%'|format(overall_rate * 100) }}
                {% else %}
                  —
                {% endif %}
              </span>
            </div>
          </div>
        </div>
      </header>

      <!-- Main grid -->
      <main class="grid gap-6 lg:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
        <!-- Left column: live + charts -->
        <section class="space-y-6">
          <!-- Live session card -->
          <div class="rounded-2xl border border-slate-800/80 bg-slate-900/70 shadow-xl shadow-sky-900/30 backdrop-blur">
            <div class="flex items-center justify-between px-4 py-3 border-b border-slate-800/80">
              <div>
                <h2 class="text-sm font-semibold tracking-wide text-slate-100">Live session snapshot</h2>
                <p class="text-xs text-slate-400 mt-1">
                  Automatically refreshes from the latest logged attempt.
                </p>
              </div>
              <span id="live-status-pill"
                    class="inline-flex items-center gap-1 rounded-full border border-sky-500/40 bg-sky-500/10 px-3 py-1 text-[11px] font-medium text-sky-100">
                <span class="h-1.5 w-1.5 rounded-full bg-sky-400 animate-ping"></span>
                <span>Waiting for first attempt…</span>
              </span>
            </div>
            <div class="grid gap-4 px-4 py-4 sm:grid-cols-3">
              <div class="space-y-1 sm:col-span-1">
                <p class="text-[11px] uppercase tracking-[0.2em] text-slate-400">Target phrase</p>
                <p id="live-phrase" class="text-sm font-medium text-slate-50 truncate">—</p>
                <p class="text-[11px] text-slate-400 mt-1">Pulled from latest row in metrics log.</p>
              </div>
              <div class="space-y-1 sm:col-span-1">
                <p class="text-[11px] uppercase tracking-[0.2em] text-slate-400">Heard</p>
                <p id="live-raw" class="text-sm font-mono text-slate-100 break-words">—</p>
              </div>
              <div class="space-y-1 sm:col-span-1">
                <p class="text-[11px] uppercase tracking-[0.2em] text-slate-400">Spoken back</p>
                <p id="live-corrected" class="text-sm font-mono text-emerald-100 break-words">—</p>
              </div>
            </div>
          </div>

          <!-- Deep Reasoning Core / CCA summary card -->
          <div class="rounded-2xl border border-indigo-800/80 bg-slate-900/70 shadow-xl shadow-indigo-900/30 backdrop-blur">
            <div class="flex items-center justify-between px-4 py-3 border-b border-indigo-800/60">
              <div>
                <h2 class="text-sm font-semibold tracking-wide text-slate-100">Deep Reasoning Core</h2>
                <p class="text-xs text-slate-400 mt-1">
                  Read-only view of Cognitive Crystal AI activity, gated by GCL and Guardian policy.
                </p>
              </div>
              <div class="text-right text-[11px]">
                {% if cca_summary.online %}
                  <span class="inline-flex items-center gap-1 rounded-full border border-emerald-400/40 bg-emerald-500/10 px-3 py-1 text-emerald-100">
                    <span class="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
                    <span>CCA Online</span>
                  </span>
                {% else %}
                  <span class="inline-flex items-center gap-1 rounded-full border border-slate-700 bg-slate-800 px-3 py-1 text-slate-300">
                    <span class="h-1.5 w-1.5 rounded-full bg-slate-500"></span>
                    <span>CCA Offline</span>
                  </span>
                {% endif %}
              </div>
            </div>
            <div class="px-4 py-4 text-xs text-slate-200 space-y-3">
              <div class="flex items-center gap-4">
                <div>
                  <p class="text-[10px] uppercase tracking-[0.2em] text-slate-400">Speech actions</p>
                  <p class="text-sm font-semibold text-slate-50">{{ cca_summary.total_speech }}</p>
                </div>
                <div>
                  <p class="text-[10px] uppercase tracking-[0.2em] text-slate-400">ABA strategies</p>
                  <p class="text-sm font-semibold text-slate-50">{{ cca_summary.total_aba }}</p>
                </div>
              </div>
              <div>
                <p class="text-[10px] uppercase tracking-[0.2em] text-slate-400 mb-1">Last CCA message</p>
                <p class="text-xs text-slate-200 italic">
                  {% if cca_summary.last_message %}
                    “{{ cca_summary.last_message }}”
                  {% else %}
                    No CCA messages yet.
                  {% endif %}
                </p>
              </div>
            </div>
          </div>

          <!-- Charts card -->
          <div class="rounded-2xl border border-slate-800/80 bg-slate-900/70 shadow-xl shadow-indigo-900/40 backdrop-blur">
            <div class="px-4 py-3 border-b border-slate-800/80 flex items-center justify-between">
              <div>
                <h2 class="text-sm font-semibold tracking-wide text-slate-100">Correction patterns</h2>
                <p class="text-xs text-slate-400 mt-1">
                  How often each phrase needs support, and how correction rate is changing over time.
                </p>
              </div>
            </div>
            <div class="px-4 pt-4 pb-5 space-y-6">
              <div class="space-y-2">
                <div class="flex items-center justify-between text-[11px] text-slate-400">
                  <span>Correction rate by phrase</span>
                  <span>Higher bars = phrase is harder right now</span>
                </div>
                <canvas id="phraseChart" class="w-full h-48"></canvas>
              </div>
              <div class="space-y-2 border-t border-slate-800/80 pt-4">
                <div class="flex items-center justify-between text-[11px] text-slate-400">
                  <span>Daily correction rate</span>
                  <span>Are sessions trending smoother over time?</span>
                </div>
                <canvas id="timelineChart" class="w-full h-40"></canvas>
              </div>
            </div>
          </div>

          <!-- Phrase table -->
          <div class="rounded-2xl border border-slate-800/80 bg-slate-950/80 shadow-lg shadow-slate-900/40 backdrop-blur">
            <div class="px-4 py-3 border-b border-slate-800/80 flex items-center justify-between">
              <div>
                <h2 class="text-sm font-semibold tracking-wide text-slate-100">Phrase difficulty map</h2>
                <p class="text-xs text-slate-400 mt-1">
                  Each phrase the child has practiced, sorted by how often it needed correction.
                </p>
              </div>
              <div class="relative">
                <input
                  id="phrase-filter"
                  type="search"
                  placeholder="Filter phrases…"
                  class="w-40 rounded-lg border border-slate-700 bg-slate-900/80 px-3 py-1.5 text-xs text-slate-100 placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-sky-500/60 focus:border-sky-400/80"
                />
              </div>
            </div>
            <div class="overflow-x-auto max-h-80">
              <table class="min-w-full text-xs text-left text-slate-200">
                <thead class="sticky top-0 bg-slate-950/95 backdrop-blur border-b border-slate-800/80">
                  <tr>
                    <th class="px-3 py-2 font-medium text-slate-400">Phrase</th>
                    <th class="px-3 py-2 font-medium text-slate-400 text-right">Attempts</th>
                    <th class="px-3 py-2 font-medium text-slate-400 text-right">Corrections</th>
                    <th class="px-3 py-2 font-medium text-slate-400 text-right">Correction rate</th>
                  </tr>
                </thead>
                <tbody id="phrase-table-body">
                  {% for row in phrases %}
                  <tr class="border-b border-slate-800/60 hover:bg-slate-900/80 transition-colors phrase-row">
                    <td class="px-3 py-2 max-w-xs truncate" data-phrase="{{ row.phrase }}">{{ row.phrase }}</td>
                    <td class="px-3 py-2 text-right tabular-nums">{{ row.attempts }}</td>
                    <td class="px-3 py-2 text-right tabular-nums">{{ row.corrections }}</td>
                    <td class="px-3 py-2 text-right tabular-nums">
                      {{ '%.0f%%'|format(row.rate * 100) }}
                    </td>
                  </tr>
                  {% endfor %}
                  {% if not phrases %}
                  <tr>
                    <td colspan="4" class="px-3 py-4 text-center text-slate-500 text-xs">
                      No sessions logged yet. Once you start practicing phrases, this table will light up.
                    </td>
                  </tr>
                  {% endif %}
                </tbody>
              </table>
            </div>
          </div>
        </section>

        <!-- Right column: strategies + guidance -->
        <section class="space-y-6">
          <!-- Strategy library -->
          <div class="rounded-2xl border border-slate-800/80 bg-slate-950/80 shadow-xl shadow-emerald-900/40 backdrop-blur">
            <div class="px-4 py-3 border-b border-slate-800/80 flex items-center justify-between">
              <div>
                <h2 class="text-sm font-semibold tracking-wide text-slate-100">Calming strategy library</h2>
                <p class="text-xs text-slate-400 mt-1">
                  Evidence-based ideas surfaced from in-the-moment cues — a quick reference for caregivers.
                </p>
              </div>
            </div>
            <div class="px-4 pt-3 pb-4 space-y-3 max-h-[18rem] overflow-y-auto">
              {% for strat in strategies %}
              <article class="rounded-xl border border-emerald-500/40 bg-emerald-500/5 px-3 py-2">
                <p class="text-[11px] uppercase tracking-[0.2em] text-emerald-300/90 mb-1">{{ strat.category }}</p>
                <h3 class="text-xs font-semibold text-emerald-50">{{ strat.title }}</h3>
                <p class="mt-1 text-[11px] leading-relaxed text-emerald-100/90">{{ strat.description }}</p>
              </article>
              {% endfor %}
              {% if not strategies %}
              <p class="text-xs text-slate-500">No strategies loaded.</p>
              {% endif %}
            </div>
          </div>

          <!-- Guidance event timeline -->
          <div class="rounded-2xl border border-slate-800/80 bg-slate-950/80 shadow-xl shadow-fuchsia-900/40 backdrop-blur">
            <div class="px-4 py-3 border-b border-slate-800/80 flex items-center justify-between">
              <div>
                <h2 class="text-sm font-semibold tracking-wide text-slate-100">Guidance timeline</h2>
                <p class="text-xs text-slate-400 mt-1">
                  When the companion stepped in — and what it said — to help regulate energy or anxiety.
                </p>
              </div>
            </div>
            <div class="px-4 pt-3 pb-4 max-h-72 overflow-y-auto">
              <ol class="space-y-3">
                {% for event in guidance_events %}
                <li class="relative pl-3">
                  <span class="absolute left-0 top-2 h-1.5 w-1.5 rounded-full bg-fuchsia-400"></span>
                  <div class="text-[11px] text-slate-400">
                    <span class="font-medium text-fuchsia-100">{{ event["event"]|replace("_", " ") }}</span>
                    <span class="mx-1 text-slate-600">•</span>
                    <span>{{ event["timestamp_iso"][:19].replace("T", " ") if event["timestamp_iso"] else "" }}</span>
                  </div>
                  <div class="text-xs font-semibold text-slate-100 mt-0.5">
                    {{ event["title"] }}
                  </div>
                  <p class="mt-0.5 text-[11px] leading-relaxed text-slate-300">
                    {{ event["message"] }}
                  </p>
                </li>
                {% endfor %}
                {% if not guidance_events %}
                <li class="text-xs text-slate-500">
                  No guidance events logged yet. When the system detects anxiety, perseveration, or high energy,
                  those calming prompts will appear here.
                </li>
                {% endif %}
              </ol>
            </div>
          </div>

          <!-- Data export hint -->
          <div class="rounded-2xl border border-slate-800/80 bg-slate-900/80 shadow-lg shadow-slate-900/40 px-4 py-3 text-[11px] text-slate-300 space-y-1.5">
            <div class="font-semibold text-slate-100 flex items-center gap-2">
              <span class="inline-flex h-5 w-5 items-center justify-center rounded-full border border-slate-700 bg-slate-800 text-[10px]">ⓘ</span>
              Therapist / research view
            </div>
            <p>
              All metrics and guidance events are mirrored as CSV files on disk
              (<code class="text-[10px] text-sky-300">metrics.csv</code> and
              <code class="text-[10px] text-sky-300">guidance.csv</code>),
              ready for import into spreadsheets or analysis tools.
            </p>
          </div>
        </section>
      </main>
    </div>

    <script>
      document.addEventListener("DOMContentLoaded", () => {
        const phraseLabels = {{ phrase_labels | tojson }};
        const phraseRates = {{ phrase_rates | tojson }};
        const timelineLabels = {{ timeline_labels | tojson }};
        const timelineRates = {{ timeline_rates | tojson }};

        const phraseCtx = document.getElementById("phraseChart");
        if (phraseCtx && phraseLabels.length) {
          new Chart(phraseCtx, {
            type: "bar",
            data: {
              labels: phraseLabels,
              datasets: [{
                label: "Correction rate (%)",
                data: phraseRates,
                borderWidth: 1
              }]
            },
            options: {
              responsive: true,
              maintainAspectRatio: false,
              plugins: {
                legend: {
                  labels: {
                    font: { size: 10 }
                  }
                }
              },
              scales: {
                x: {
                  ticks: {
                    maxRotation: 45,
                    minRotation: 0,
                    autoSkip: true,
                    font: { size: 9 }
                  },
                  grid: { display: false }
                },
                y: {
                  min: 0,
                  max: 100,
                  ticks: {
                    stepSize: 20,
                    font: { size: 9 },
                    callback: (value) => value + "%"
                  },
                  grid: {
                    borderDash: [4, 4]
                  }
                }
              }
            }
          });
        }

        const timelineCtx = document.getElementById("timelineChart");
        if (timelineCtx && timelineLabels.length) {
          new Chart(timelineCtx, {
            type: "line",
            data: {
              labels: timelineLabels,
              datasets: [{
                label: "Daily correction rate (%)",
                data: timelineRates,
                tension: 0.35,
                pointRadius: 2.5,
                borderWidth: 2
              }]
            },
            options: {
              responsive: true,
              maintainAspectRatio: false,
              plugins: {
                legend: {
                  labels: { font: { size: 10 } }
                }
              },
              scales: {
                x: {
                  ticks: { font: { size: 9 } },
                  grid: { display: false }
                },
                y: {
                  min: 0,
                  max: 100,
                  ticks: {
                    stepSize: 20,
                    font: { size: 9 },
                    callback: (value) => value + "%"
                  },
                  grid: { borderDash: [4, 4] }
                }
              }
            }
          });
        }

        // Phrase filter
        const filterInput = document.getElementById("phrase-filter");
        if (filterInput) {
          filterInput.addEventListener("input", (e) => {
            const q = e.target.value.toLowerCase();
            document.querySelectorAll(".phrase-row").forEach((row) => {
              const cell = row.querySelector("[data-phrase]");
              const text = (cell?.dataset.phrase || "").toLowerCase();
              row.style.display = text.includes(q) ? "" : "none";
            });
          });
        }

        // Live session updater: poll metrics API for most recent row
        async function refreshLive() {
          try {
            const res = await fetch("/api/metrics");
            if (!res.ok) return;
            const data = await res.json();
            if (!Array.isArray(data) || data.length === 0) return;

            const last = data[data.length - 1];
            const phrase = last.phrase_text || last.phrase_id || "—";
            const raw = last.raw_text || "—";
            const corrected = last.corrected_text || "—";

            const needsCorrection =
              last.needs_correction === "1" ||
              last.needs_correction === 1 ||
              last.needs_correction === true;

            const phraseEl = document.getElementById("live-phrase");
            const rawEl = document.getElementById("live-raw");
            const correctedEl = document.getElementById("live-corrected");
            const pill = document.getElementById("live-status-pill");

            if (phraseEl) phraseEl.textContent = phrase;
            if (rawEl) rawEl.textContent = raw;
            if (correctedEl) correctedEl.textContent = corrected;
            if (pill) {
              pill.querySelector("span:nth-child(2)").textContent =
                needsCorrection ? "Correction suggested" : "Sounded great";
            }
          } catch (err) {
            // fail silently in UI
          }
        }

        refreshLive();
        setInterval(refreshLive, 5000);
      });
    </script>
  </body>
</html>"""


def _load_rows(csv_path) -> List[Dict[str, Any]]:
    if not csv_path.exists():
        return []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        return list(reader)


def _load_guidance(csv_path) -> List[Dict[str, Any]]:
    if not csv_path.exists():
        return []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        return list(reader)


def _voice_profile_counts(config: CompanionConfig) -> Dict[str, int]:
    base = config.paths.voices_dir / "voice_profile"
    counts: Dict[str, int] = {}
    for style in ("neutral", "calm", "excited"):
        style_dir = base / style
        counts[style] = len(list(style_dir.glob("*.wav"))) if style_dir.exists() else 0
    return counts


def _recent_behavior_events(config: CompanionConfig, limit: int = 50) -> List[Dict[str, Any]]:
    rows = _load_guidance(config.paths.guidance_csv)
    if not rows:
        return []
    return rows[-limit:]


def _cca_summary(config: CompanionConfig) -> Dict[str, Any]:
    """Summarize Deep Reasoning Core / CCA activity from guidance log."""
    rows = _load_guidance(config.paths.guidance_csv)
    total_cca_speech = 0
    total_cca_aba = 0
    last_cca_message = ""

    for row in rows:
        event = (row.get("event") or "").strip()
        if event == "cca_speech":
            total_cca_speech += 1
            last_cca_message = row.get("message") or last_cca_message
        elif event == "cca_aba":
            total_cca_aba += 1
            last_cca_message = row.get("message") or last_cca_message

    return {
        "online": bool(total_cca_speech or total_cca_aba),
        "total_speech": total_cca_speech,
        "total_aba": total_cca_aba,
        "last_message": last_cca_message,
    }


def _summarize_metrics(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str], List[float]]:
    """Return (phrase_rows, timeline_labels, timeline_rates)."""
    phrase_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"phrase": "Unknown", "attempts": 0, "corrections": 0})
    daily: Dict[str, Dict[str, int]] = defaultdict(lambda: {"attempts": 0, "corrections": 0})

    for row in rows:
        pid = row.get("phrase_id") or "Unknown"
        phrase_text = row.get("phrase_text") or pid
        needs_correction = (row.get("needs_correction") == "1")

        pstats = phrase_stats[pid]
        pstats["phrase"] = phrase_text
        pstats["attempts"] += 1
        if needs_correction:
            pstats["corrections"] += 1

        ts = row.get("timestamp_iso") or ""
        if ts:
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                date_key = dt.date().isoformat()
            except Exception:
                date_key = ts.split("T", 1)[0]
            dstats = daily[date_key]
            dstats["attempts"] += 1
            if needs_correction:
                dstats["corrections"] += 1

    phrase_rows: List[Dict[str, Any]] = []
    total_attempts = 0
    total_corrections = 0

    for stats in phrase_stats.values():
        attempts = stats["attempts"]
        corrections = stats["corrections"]
        rate = (corrections / attempts) if attempts else 0.0
        phrase_rows.append(
            {
                "phrase": stats["phrase"],
                "attempts": attempts,
                "corrections": corrections,
                "rate": rate,
            }
        )
        total_attempts += attempts
        total_corrections += corrections

    phrase_rows.sort(key=lambda r: r["attempts"], reverse=True)

    timeline_labels: List[str] = []
    timeline_rates: List[float] = []
    for date_key in sorted(daily.keys()):
        attempts = daily[date_key]["attempts"]
        corrections = daily[date_key]["corrections"]
        rate = (corrections / attempts) if attempts else 0.0
        timeline_labels.append(date_key)
        timeline_rates.append(rate * 100.0)

    return phrase_rows, timeline_labels, timeline_rates


def create_app(config: CompanionConfig = CONFIG, settings_store: SettingsStore | None = None) -> Flask:
    app = Flask(__name__)
    policy_path = config.paths.root / "guardian_policy.json"

    @app.get("/api/metrics")
    def metrics_api() -> Any:
        rows = _load_rows(config.paths.metrics_csv)
        return jsonify(rows)

    @app.get("/api/voice-profile")
    def voice_profile_api() -> Any:
        return jsonify(_voice_profile_counts(config))

    @app.get("/api/strategies")
    def strategies_api() -> Any:
        return jsonify(
            [
                {"category": s.category, "title": s.title, "description": s.description}
                for s in STRATEGIES
            ]
        )

    @app.get("/api/behavior")
    def behavior_api() -> Any:
        return jsonify(_recent_behavior_events(config))

    @app.get("/api/guidance-events")
    def guidance_events_api() -> Any:
        rows = _load_guidance(config.paths.guidance_csv)
        return jsonify(rows)

    @app.get("/api/settings")
    def settings_api() -> Any:
        if settings_store is None:
            return jsonify({})
        return jsonify(settings_store.get_settings())

    @app.patch("/api/settings")
    def settings_update() -> Any:
        if settings_store is None:
            return jsonify({"error": "settings store unavailable"}), 400
        payload = request.get_json(force=True, silent=True) or {}
        settings_store.update(
            correction_echo_enabled=payload.get("correction_echo_enabled"),
            support_voice_enabled=payload.get("support_voice_enabled"),
        )
        config.behavior.correction_echo_enabled = bool(settings_store.data.get("correction_echo_enabled", True))
        config.behavior.support_voice_enabled = bool(settings_store.data.get("support_voice_enabled", False))
        return jsonify(settings_store.get_settings())

    @app.get("/api/support-phrases")
    def support_phrases_api() -> Any:
        if settings_store is None:
            return jsonify([])
        return jsonify(settings_store.list_support_phrases())

    @app.post("/api/support-phrases")
    def add_support_phrase() -> Any:
        if settings_store is None:
            return jsonify({"error": "settings store unavailable"}), 400
        payload = request.get_json(force=True, silent=True) or {}
        phrase = payload.get("phrase")
        if not phrase:
            return jsonify({"error": "phrase missing"}), 400
        settings_store.add_support_phrase(phrase)
        return jsonify(settings_store.list_support_phrases())

    @app.get("/api/guardian-policy")
    def guardian_policy_get() -> Any:
        """Return current GuardianPolicy JSON (if present)."""
        if policy_path.exists():
            policy = GuardianPolicy.load(policy_path)
            return jsonify(policy.raw)
        # If no file, return the default bootstrapped policy
        policy = GuardianPolicy.load(policy_path)
        return jsonify(policy.raw)

    @app.put("/api/guardian-policy")
    def guardian_policy_put() -> Any:
        """Persist a new GuardianPolicy JSON document."""
        payload = request.get_json(force=True, silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "invalid policy payload"}), 400
        policy = GuardianPolicy(raw=payload)
        policy.save(policy_path)
        return jsonify({"status": "ok"})

    @app.post("/record_facet")
    def record_facet() -> Any:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        if 'style' not in request.form:
            return jsonify({"error": "No style provided"}), 400

        audio_file = request.files['audio']
        style = request.form['style']

        if not audio_file.filename:
            return jsonify({"error": "No selected file"}), 400

        try:
            temp_dir = config.paths.voices_dir / "temp_facets"
            temp_dir.mkdir(exist_ok=True)
            temp_wav_path = temp_dir / f"{style}_{os.urandom(4).hex()}.wav"
            audio_file.save(temp_wav_path)

            data, sr = sf.read(temp_wav_path, dtype="float32")
            if sr != config.audio.sample_rate:
                data = librosa.resample(data, orig_sr=sr, target_sr=config.audio.sample_rate)

            voice_profile = VoiceProfile(audio=config.audio, base_dir=config.paths.voices_dir / "voice_profile")
            voice_profile.add_sample_from_wav(np.asarray(data, dtype=np.float32), style)

            os.remove(temp_wav_path)

            return jsonify({"status": "success", "message": f"Facet '{style}' recorded successfully."}), 200
        except Exception as e:
            print(f"Error recording facet: {e}")
            return jsonify({"error": str(e)}), 500

    @app.get("/")
    def index() -> str:
        rows = _load_rows(config.paths.metrics_csv)
        phrase_rows, timeline_labels, timeline_rates = _summarize_metrics(rows)
        guidance_rows = _load_guidance(config.paths.guidance_csv)

        recent_guidance = guidance_rows[-25:]

        cca_summary = _cca_summary(config)

        total_attempts = sum(r["attempts"] for r in phrase_rows)
        total_corrections = sum(r["corrections"] for r in phrase_rows)
        overall_rate = (total_corrections / total_attempts) if total_attempts else 0.0

        featured_strategies = STRATEGIES[:8]

        return render_template_string(
            TEMPLATE,
            child_name=config.child_name,
            total_attempts=total_attempts,
            overall_rate=overall_rate,
            phrases=phrase_rows,
            phrase_labels=[r["phrase"] for r in phrase_rows],
            phrase_rates=[round(r["rate"] * 100.0, 1) for r in phrase_rows],
            timeline_labels=timeline_labels,
            timeline_rates=[round(v, 1) for v in timeline_rates],
            strategies=featured_strategies,
            guidance_events=recent_guidance,
            cca_summary=cca_summary,
        )

    return app


def main() -> None:
    app = create_app()
    app.run(host="0.0.0.0", port=8765, debug=False)


if __name__ == "__main__":
    main()
