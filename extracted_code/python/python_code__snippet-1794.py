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


