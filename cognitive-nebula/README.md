<div align="center">
<img width="1200" height="475" alt="GHBanner" src="https://github.com/user-attachments/assets/0aa67016-6eaf-458a-adb2-6e31a0763ed6" />
</div>

# Run and deploy your AI Studio app

This contains everything you need to run your app locally.

View your app in AI Studio: https://ai.studio/apps/drive/1nHSaCwo_Tk2flLFlTdn2ikDHRkY2JxY7

## Run Locally

**Prerequisites:**  Node.js and [Ollama](https://ollama.com/) running locally (default http://localhost:11434). Any open-source chat model available to Ollama will work; defaults to `llama3`.
For local image generation, run [AUTOMATIC1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui) with the API enabled at `http://localhost:7860` (configurable in settings). If local SD is unreachable, the app will fall back to Pollinations (internet required).


1. Install dependencies:
   `npm install`
2. (Optional) Set `VITE_OLLAMA_URL` and `VITE_OLLAMA_TEXT_MODEL` in `.env.local` to point at a different Ollama host or model.
3. (Optional) Set `VITE_SD_URL` if your AUTOMATIC1111 host differs from `http://localhost:7860`.
4. Run the app:
   `npm run dev`

## Run as a desktop app

1. Install desktop runtime deps: `npm install` (adds Electron).
2. Build the web bundle: `npm run build`
3. Launch the desktop window: `npm run desktop`

Tips:
- Electron will try to auto-start `ollama serve` if it is installed but not running; otherwise the app will prompt you to start it.
- Electron will also try to auto-start Automatic1111 if you set `SD_WEBUI_CMD` (e.g., the full command you use to launch it). Otherwise, it will just warn you to start it manually.
- Desktop env vars: `SD_URL` (host for Automatic1111), `SD_WEBUI_CMD` (optional auto-start command).
- For live reload during development, run `npm run dev` in one terminal and in another run: `VITE_DEV_SERVER_URL=http://localhost:5173 electron .`
- To change the window icon, place a PNG at `icons/app.png`.

### One-click icon on Linux
1. Copy `CognitiveNebula.desktop` to `~/.local/share/applications/`
2. Make it executable: `chmod +x ~/.local/share/applications/CognitiveNebula.desktop`
3. It should appear in your app launcher. The `.desktop` file points to `npm run desktop` inside this project path; adjust the paths in that file if you move the repo.
