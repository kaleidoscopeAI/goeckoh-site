# Repository Guidelines

## Project Structure & Module Organization

- Core Python runtime and speech companion logic live in root-level `*.py` modules (for example `main.py`, `core.py`, `speechinterventionsystem_2.py`, `echo_gui.py`).
- The React/Vite visualizer lives alongside them (`App.jsx`, `main.jsx`, `vite.config.js`, `simulationService.js`).
- Python dependencies are defined in `requirements.txt`; long-form design and research documents are under `documentsforsystem/` (treat these as read-only reference).
- Build artefacts such as `__pycache__/`, `speechinterventionsystem`, `*.spec`, and `*.db` files should not be edited or extended manually.

## Build, Test, and Development Commands

- `python3 main.py` – run the default Organic AI / EchoVoice demo (as described in `README.md`).
- `python3 echo_gui.py` or `python3 tk_gui.py` – launch experimental desktop dashboards.
- `python3 -m venv .venv && pip install -r requirements.txt` – create and populate a local Python dev environment.
- `npm install` – install front-end dependencies; `npm run dev` – start the Vite dev server; `npm run build` – build production assets.

## Coding Style & Naming Conventions

- Python: 4-space indentation, PEP 8 layout, `snake_case` for functions/variables, `PascalCase` for classes; prefer type hints and short docstrings on public APIs.
- JS/TS/React: ES modules with `import`/`export`, `const`/`let`; components in `PascalCase` (`HUD`, `FlowGraph`), hooks as `useCamelCase`.
- Keep modules cohesive with one primary responsibility per file; prefer small helpers over long monolithic functions.

## Testing Guidelines

- There is no central automated test suite yet; when adding behavior, include focused unit tests where practical.
- For Python, prefer `pytest` or `unittest` with files named `test_<module>.py`, either beside the module or under a future `tests/` directory.
- For React, co-locate `*.test.jsx/tsx` next to components and cover key interaction paths.

## Commit & Pull Request Guidelines

- Use small, focused commits with imperative, present-tense subjects (for example `core: refine energetics logging`, `ui: improve node hover state`).
- Conventional-Commit prefixes like `feat:`, `fix:`, and `chore:` are welcome but not required.
- Pull requests (or equivalent review units) should explain motivation, key changes, and how to run/verify them; link to relevant documents in `documentsforsystem/` and include screenshots or short clips for UI-facing changes.

## Security & Configuration Tips

- Do not commit secrets, API keys, or real user data; configure keys via environment variables (for example `GEMINI_API_KEY` for the Vite front-end).
- Treat `onbrain*.db` and other `.db` files as local data stores; avoid adding new binary data artefacts to version control.

## Agent-Specific Instructions

- Automated tools and coding agents should prefer minimal, targeted patches, avoid editing large narrative files under `documentsforsystem/` unless explicitly requested, and never modify compiled binaries (`speechinterventionsystem`, `.pyz`, `.spec`) except when packaging work is the explicit goal.

