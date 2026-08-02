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
