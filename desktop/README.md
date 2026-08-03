# Goeckoh Desktop (Electron)

A real, tested offline shell around the same correction engine that runs at
`goeckoh.com` — `app/index.html` is a bundled copy of
`real_time_therapeutic_voice_cloning_system.html`, DSP unmodified.

## What this actually adds over the web app

- **Real device-bound license activation** (`/license/activate`), cached
  locally so the app keeps working with **no internet connection** after
  first activation — see `ensureActivated()` in `main.js`.
- **Runs in the background**, not tied to a visible window: closing the
  window hides it to the system tray instead of quitting; `--hidden` launch
  flag (used automatically by "Start at login") skips showing a window at
  all; correction starts on its own (`quickStart()` auto-triggered) instead
  of requiring a click every launch.
- **Real mic permission grant** — Electron denies `getUserMedia` by default;
  without the permission handler in `main.js`, the app would silently fail
  for every user on first use.
- **Working local progress backend** (`local-backend.js`, port 8000) — the
  bundled page already has `isLocal` / `localhost:8000` branches for
  guardian relay and session stats; nothing previously ran anything there.
  This implements it for real: session metrics (F0, HNR, latency, voiced
  ratio, correction count) logged locally as they happen and served back
  via `/session/stats`, plus the ephemeral live-relay endpoints
  (`/session/new-code`, `/ws/broadcast/:code`, `/ws/monitor/:code`) ported
  faithfully from `backend/main.py`'s design — same no-storage relay, just
  running on-device instead of nonexistent. Consistent with the product's
  stated architecture (`backend/models.py`'s `User` docstring: "no voice
  data, no session metrics, no PHI... all therapeutic data lives on the
  user's device") — nothing here is sent anywhere.
  `/session/aba-progress` honestly returns `not_implemented` rather than
  fabricating skill-mastery numbers — that needs a real skill-prompt UI
  that doesn't exist in this build yet, not guessed statistics.

## Verified, not assumed

Everything above was actually run and measured (`test/electron_*_diag.js`,
Playwright's Electron support, real fake-audio input via Chromium's
`--use-file-for-fake-audio-capture`, not silence or a pure tone):

- Cached-license offline fallback: confirmed it loads the real app (not the
  activation screen) with the network entirely blocked.
- Output audio quality: 0% clipped samples, no NaN/instability, on a
  synthesized voice-like signal — see the commit history for before/after
  numbers on the related `demo.html` clipping fix, same verification method.
- `--hidden` launch: window `isVisible()` confirmed `false`.
- Auto-start: Quick Start button reaches "✓ Running" with zero clicks.
- Close-to-tray: after `window.close()`, the process is still alive and the
  window still exists (just hidden) — not destroyed, not quit.
- Local progress backend: ran a real 6-second session with synthesized
  voice input, then queried `/session/stats` directly — returned real
  computed numbers (F0 in the raw log matched the synthesized signal's
  actual 110Hz pitch exactly), not placeholders. Confirmed the live relay
  end-to-end (broadcaster message received by a connected monitor through
  `/ws/broadcast` → `/ws/monitor`). Confirmed `/session/aba-progress`
  returns an honest `not_implemented` instead of fake data.

## Known limitations — stated plainly, not glossed over

- **Only the core correction tool is bundled.** The nav bar still links to
  `advanced_voice_visualizer_pro.html`, `voice_therapy_profile_transform.html`,
  `goeckoh_dashboard.html`, `voice_game.html`, `voice_setup.html` — none of
  those are bundled yet, so those links won't resolve offline.
- **`powerSaveBlocker` prevents idle/display sleep, not a closed laptop
  lid.** Most OSes suspend on lid-close regardless of what a background app
  requests unless the system's own power settings say otherwise. "Runs
  effortlessly in the background" means minimized/tray/no-window-open, not
  "survives the lid being shut" — that's an OS setting, not an app one.
  Users who need lid-closed operation need to change their OS power
  settings; the app can't override that.
- **Only built and tested for Linux in this environment.** The
  `electron-builder` config below is written for all three platforms, but
  macOS/Windows packages need to actually be built and smoke-tested on
  those platforms (or via the CI workflow) before being trusted — this
  sandbox can't cross-compile-and-verify them the way it did for Linux.
- **Google Fonts and the Guardian relay WebSocket both fail gracefully
  offline** (already designed that way in the original page) — cosmetic
  font fallback and no remote monitoring, not a functional break.
- **The live relay only works when the guardian's device can reach
  `ws://<patient-device-ip>:8000` directly** — same local network / VPN,
  same as the original design. This is not a remote-guardian cloud feature
  and isn't meant to become one without a real, separate conversation about
  whether any cross-device sync should exist given the product's current
  "nothing leaves the device" privacy design.
- **`/session/aba-progress` is genuinely not implemented**, not just
  incomplete — there is no UI anywhere in this app for prompting a skill
  attempt and recording success/fail, which is what that data would
  actually require. Building it means designing that UI first, not just
  wiring an endpoint.
- **The local session log grows unbounded** (`sessions/session_log.jsonl`)
  — fine for MVP verification, worth adding rotation/pruning before this
  sees real extended daily use.

## Building

```bash
cd desktop
npm install
npm start                  # run directly
npx electron-builder --linux --mac --win   # package (needs per-platform toolchains/signing for mac/win)
```

## Replacing the old manually-uploaded binaries

The binaries previously in `kaleidoscopeai/goeckoh-releases` v1.0.0 had no
source or build process anywhere in this repo — this directory is that
missing source. A CI workflow (matching `android/`'s pattern) should build
and publish real, reproducible releases from here going forward instead of
manual uploads.
