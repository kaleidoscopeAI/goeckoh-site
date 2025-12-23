# run_gui.py
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

# Ensure repo root on sys.path
# Repo root is two levels up from this file
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Optionally build Rust FFI before launching GUI:
try:
    from goeckoh.apps.run_exocortex import build_rust_kernel  # type: ignore
except Exception:
    build_rust_kernel = None

from goeckoh_gui.theme import apply_palette, QSS
from goeckoh_gui.controllers.main_controller import MainController
from goeckoh_gui.views.main_window import GoeckohMainWindow


def main():
    if build_rust_kernel is not None:
        build_rust_kernel()

    app = QApplication(sys.argv)
    apply_palette(app)
    app.setStyleSheet(QSS)

    controller = MainController()
    window = GoeckohMainWindow(controller)
    window.resize(1100, 700)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
