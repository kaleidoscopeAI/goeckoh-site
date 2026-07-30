# Goeckoh — Android app (Trusted Web Activity)

This wraps the live `goeckoh.com` voice-correction tool
(`real_time_therapeutic_voice_cloning_system.html`) as an installable Android
app using a [Trusted Web Activity](https://developer.chrome.com/docs/android/trusted-web-activity/) —
Chrome renders the real site full-screen with no browser chrome. The website
itself is untouched; the only site-side addition is
`goeckoh-site/.well-known/assetlinks.json`, a verification file required for
Android to trust this app with the domain (no visible page changed).

Because the app is just a thin, verified pointer at the live site, any change
you make to the website ships to app users immediately — there is no separate
app codebase to keep in sync.

- Package name: `com.goeckoh.voicecorrection`
- Launch URL: `https://goeckoh.com/real_time_therapeutic_voice_cloning_system.html`
- Min/target/compile SDK: 21 / 36 / 36

## Why it can't be built in this sandbox

Building an Android app requires the Android Gradle Plugin, `androidx.*`
libraries, and the SDK platform/build-tools — all hosted on
`dl.google.com`. This session's network policy blocks that host, so the
build must happen somewhere with normal internet access: your own machine,
Android Studio, or CI (see the GitHub Actions workflow below, which builds
automatically).

## Building locally

Requires: JDK 17+, Android SDK (Android Studio installs this for you).

```bash
cd android
./gradlew :app:bundleRelease     # -> app/build/outputs/bundle/release/app-release.aab
./gradlew :app:assembleRelease   # -> app/build/outputs/apk/release/app-release.apk
```

Without signing env vars set (see below), the release build is unsigned —
fine for local testing, not for Play Store upload.

## Building via GitHub Actions (recommended)

`.github/workflows/android-release.yml` builds the AAB/APK on every push to
`android/**` on `main`, and can be run manually from the Actions tab
("Run workflow"). It needs these repo secrets to produce a **signed** build
(without them it still builds, just unsigned):

| Secret | Value |
|---|---|
| `ANDROID_KEYSTORE_BASE64` | `base64 -w0 your-keystore.jks` |
| `ANDROID_KEYSTORE_PASSWORD` | keystore password |
| `ANDROID_KEY_ALIAS` | key alias inside the keystore |
| `ANDROID_KEY_PASSWORD` | key password (same as store password for PKCS12 keystores) |

The signed `.aab` is what you upload to Play Console.

## Signing key

**Whoever holds this keystore controls all future updates to this app —
losing it (before enrolling in Play App Signing) permanently locks you out of
updating `com.goeckoh.voicecorrection`.** Back it up somewhere durable
(password manager, offline drive) the moment you receive it. It must never be
committed to this repo (`.gitignore` already excludes `*.jks`/`*.keystore`).

On first upload to Play Console, opt into **Play App Signing** — Google then
holds the real distribution signing key and this keystore becomes just your
"upload key," which is easier to rotate if ever compromised.

After enrolling, Play Console shows an **App signing key certificate** (under
Setup → App integrity) with its own SHA-256 fingerprint — add that fingerprint
as a second entry in `goeckoh-site/.well-known/assetlinks.json`'s
`sha256_cert_fingerprints` array alongside the upload key's, so the app keeps
working after Google re-signs it for distribution.

## Icons

Generated from `goeckoh-site/images/brand-sheet.png` (the dark-background
mark). Regenerate by re-running the crop/resize steps against that source
image if the brand mark changes.
