"""Legacy entrypoint kept for backwards compatibility."""

from __future__ import annotations

try:
    from cli import main
except Exception:  # pragma: no cover - package-style usage
    try:
        from .cli import main  # type: ignore
    except Exception as exc:
        raise ImportError("cli module is unavailable") from exc


if __name__ == "__main__":
    main()
