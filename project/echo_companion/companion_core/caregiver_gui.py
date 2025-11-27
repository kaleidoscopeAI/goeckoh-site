#!/usr/bin/env python3
"""
Native caregiver console for Jackson's Companion.
Reads local API at http://127.0.0.1:8081 and CSV logs under logs/.
No browser, no Streamlit.
"""

import csv
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import requests
import sounddevice as sd
import soundfile as sf
from PyQt6.QtCore import Qt, QTimer, QThread, QUrl, pyqtSignal
from PyQt6.QtGui import QFont, QColor, QDesktopServices
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QProgressBar,
    QGroupBox,
    QTableWidget,
    QTableWidgetItem,
    QMessageBox,
    QHeaderView,
)

API_BASE = "http://127.0.0.1:8081"
LOGS_DIR = "logs"
VOICE_SAMPLES_DIR = os.path.expanduser(os.path.join("~", "JacksonCompanion", "voice_samples"))
os.makedirs(VOICE_SAMPLES_DIR, exist_ok=True)


class CloneRecorderWorker(QThread):
    """
    Background worker that records a short WAV clip to the speaker_ref_dir
    used by VoiceCrystal (voice_ref/jackson under this file's directory).
    """

    finished = pyqtSignal(str, bool, str)  # (path, success, error_message)

    def __init__(self, seconds: int = 5, parent=None):
        super().__init__(parent)
        self.seconds = seconds

    def run(self) -> None:
        try:
            samplerate = 16000
            channels = 1

            data = sd.rec(
                int(self.seconds * samplerate),
                samplerate=samplerate,
                channels=channels,
                dtype="float32",
            )
            sd.wait()

            base_dir = Path(__file__).resolve().parent
            ref_dir = base_dir / "voice_ref" / "jackson"
            ref_dir.mkdir(parents=True, exist_ok=True)

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = ref_dir / f"jackson_ref_{ts}.wav"

            sf.write(out_path, data, samplerate)
            self.finished.emit(str(out_path), True, "")
        except Exception as e:
            self.finished.emit("", False, str(e))


def get_json(path: str) -> Dict[str, Any] | None:
    try:
        r = requests.get(API_BASE + path, timeout=1.0)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def post_json(path: str) -> Dict[str, Any] | None:
    try:
        r = requests.post(API_BASE + path, timeout=2.0)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


class CaregiverWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Jackson's Companion — Caregiver Console")
        self.resize(1100, 800)
        self._init_ui()
        self._apply_style()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(1000)
        self.refresh()

    def _init_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 8, 10, 8)
        root.setSpacing(8)

        # Top controls
        ctrl = QGroupBox("System Control")
        ctrl_layout = QHBoxLayout(ctrl)
        self.health_label = QLabel("Health: unknown")
        bold = QFont()
        bold.setBold(True)
        self.health_label.setFont(bold)
        ctrl_layout.addWidget(self.health_label)

        self.kill_btn = QPushButton("Kill (/kill)")
        self.kill_btn.clicked.connect(self.on_kill)
        ctrl_layout.addWidget(self.kill_btn)

        self.wipe_btn = QPushButton("Wipe (/wipe)")
        self.wipe_btn.clicked.connect(self.on_wipe)
        ctrl_layout.addWidget(self.wipe_btn)
        ctrl_layout.addStretch(1)
        root.addWidget(ctrl)

        # Metrics
        metrics = QGroupBox("Behavior & Metrics")
        mlay = QHBoxLayout(metrics)
        self.gcl_bar = QProgressBar()
        self.gcl_bar.setRange(0, 1000)
        self.gcl_label = QLabel("GCL: 0.000")
        gcl_box = QVBoxLayout()
        gcl_box.addWidget(self.gcl_label)
        gcl_box.addWidget(self.gcl_bar)
        mlay.addLayout(gcl_box)

        self.risk_bar = QProgressBar()
        self.risk_bar.setRange(0, 100)
        self.risk_label = QLabel("Meltdown risk: 0%")
        risk_box = QVBoxLayout()
        risk_box.addWidget(self.risk_label)
        risk_box.addWidget(self.risk_bar)
        mlay.addLayout(risk_box)

        self.style_label = QLabel("Style: N/A")
        mlay.addWidget(self.style_label)

        self.attempts_label = QLabel("Attempts: 0")
        self.sim_label = QLabel("Avg similarity: 0.00")
        mlay.addWidget(self.attempts_label)
        mlay.addWidget(self.sim_label)
        root.addWidget(metrics)

        # Voice profile
        voice_box = QGroupBox("Voice usage counts")
        vlay = QHBoxLayout(voice_box)
        self.voice_labels: Dict[str, QLabel] = {}
        for style in ("calm", "inner", "excited"):
            lbl = QLabel(f"{style}: 0")
            self.voice_labels[style] = lbl
            vlay.addWidget(lbl)
        root.addWidget(voice_box)

        # Clone voice samples (for XTTS voice cloning)
        clone_box = QGroupBox("Clone voice samples")
        clone_layout = QHBoxLayout(clone_box)

        left = QVBoxLayout()
        self.clone_count_label = QLabel("Samples: 0 files")
        bold_small = QFont()
        bold_small.setBold(True)
        self.clone_count_label.setFont(bold_small)
        left.addWidget(self.clone_count_label)

        self.record_btn = QPushButton("Record new sample")
        self.record_btn.clicked.connect(self.on_record_clone_sample)
        left.addWidget(self.record_btn)

        self.open_folder_btn = QPushButton("Open samples folder")
        self.open_folder_btn.clicked.connect(self.on_open_clone_folder)
        left.addWidget(self.open_folder_btn)

        self.refresh_samples_btn = QPushButton("Refresh list")
        self.refresh_samples_btn.clicked.connect(self._refresh_clone_samples)
        left.addWidget(self.refresh_samples_btn)

        left.addStretch(1)
        clone_layout.addLayout(left)

        self.clone_table = QTableWidget(0, 3)
        self.clone_table.setHorizontalHeaderLabels(["File", "Duration (s)", "Path"])
        header = self.clone_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.clone_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.clone_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.clone_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        clone_layout.addWidget(self.clone_table)

        root.addWidget(clone_box)

        # Voice clone reference clips
        clone_group = QGroupBox("Voice Clone Reference Clips")
        clone_layout = QVBoxLayout()

        clone_label = QLabel(
            "Press 'Record 5s Test Clip' and have Jackson speak naturally.\n"
            "Each press saves a new WAV file into the clone directory used by the system."
        )
        clone_label.setWordWrap(True)
        clone_layout.addWidget(clone_label)

        self.clone_status_label = QLabel("No reference clips recorded yet.")
        clone_layout.addWidget(self.clone_status_label)

        self.clone_record_button = QPushButton("Record 5s Test Clip")
        self.clone_record_button.clicked.connect(self._on_record_clone_clip)
        clone_layout.addWidget(self.clone_record_button)

        clone_group.setLayout(clone_layout)
        root.addWidget(clone_group)

        # Logs
        log_box = QGroupBox("Recent events")
        llay = QHBoxLayout(log_box)
        self.attempts_table = QTableWidget(0, 5)
        self.attempts_table.setHorizontalHeaderLabels(["ts", "raw", "corrected", "dtw", "success"])
        self.attempts_table.horizontalHeader().setStretchLastSection(True)
        self.attempts_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        llay.addWidget(self.attempts_table)

        self.guidance_table = QTableWidget(0, 4)
        self.guidance_table.setHorizontalHeaderLabels(["ts", "event", "phrase", "risk"])
        self.guidance_table.horizontalHeader().setStretchLastSection(True)
        self.guidance_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        llay.addWidget(self.guidance_table)
        root.addWidget(log_box, 1)

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background-color: #121212; color: #f0f0f0; }
            QGroupBox { border: 1px solid #333; border-radius: 6px; margin-top: 8px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #aaa; }
            QPushButton { background-color: #333; border-radius: 4px; padding: 6px 10px; }
            QPushButton:hover { background-color: #444; }
            """
        )

    def refresh(self) -> None:
        self._refresh_health()
        self._refresh_behavior()
        self._refresh_voice_profile()
        self._refresh_logs()
        self._refresh_clone_samples()
        self._refresh_clone_samples()

    def _refresh_health(self) -> None:
        data = get_json("/health")
        healthy = bool(data and data.get("status") == "healthy")
        self.health_label.setText(f"Health: {'healthy' if healthy else 'offline'}")
        pal = self.health_label.palette()
        pal.setColor(self.health_label.foregroundRole(), QColor("#7CFC00" if healthy else "#ff6666"))
        self.health_label.setPalette(pal)

    def _refresh_behavior(self) -> None:
        behavior = get_json("/api/behavior") or {}
        metrics = get_json("/api/metrics") or {}
        gcl = float(behavior.get("gcl", 0.0))
        risk = float(behavior.get("risk", 0.0))
        style = behavior.get("style", "inner")
        attempts = int(metrics.get("attempts", metrics.get("attempts_total", 0)))
        avg_sim = float(metrics.get("avg_similarity", 0.0))

        self.gcl_label.setText(f"GCL: {gcl:.3f}")
        self.gcl_bar.setValue(int(gcl * 1000))
        self.risk_label.setText(f"Meltdown risk: {risk*100:.1f}%")
        self.risk_bar.setValue(int(risk * 100))
        self.style_label.setText(f"Style: {style}")
        self.attempts_label.setText(f"Attempts: {attempts}")
        self.sim_label.setText(f"Avg similarity: {avg_sim:.2f}")

    def _refresh_voice_profile(self) -> None:
        data = get_json("/api/voice-profile") or {}
        for style, lbl in self.voice_labels.items():
            lbl.setText(f"{style}: {int(data.get(style, 0))}")

    def _refresh_logs(self) -> None:
        attempts_path = os.path.join(LOGS_DIR, "attempts.csv")
        if os.path.exists(attempts_path):
            try:
                with open(attempts_path, "r", newline="", encoding="utf-8") as f:
                    rows = list(csv.reader(f))
                if len(rows) > 1:
                    self._populate(self.attempts_table, rows[1:][-20:])
            except Exception:
                pass
        guidance_path = os.path.join(LOGS_DIR, "guidance.csv")
        if os.path.exists(guidance_path):
            try:
                with open(guidance_path, "r", newline="", encoding="utf-8") as f:
                    rows = list(csv.reader(f))
                if len(rows) > 1:
                    self._populate(self.guidance_table, rows[1:][-20:])
            except Exception:
                pass

    def _refresh_clone_samples(self) -> None:
        """Scan VOICE_SAMPLES_DIR for .wav files and show them in the table."""
        files: List[tuple[str, float, str]] = []

        if os.path.isdir(VOICE_SAMPLES_DIR):
            for name in sorted(os.listdir(VOICE_SAMPLES_DIR)):
                if not name.lower().endswith(".wav"):
                    continue
                path = os.path.join(VOICE_SAMPLES_DIR, name)
                duration = 0.0
                try:
                    info = sf.info(path)
                    if info.samplerate > 0:
                        duration = info.frames / float(info.samplerate)
                except Exception:
                    pass
                files.append((name, duration, path))

        count = len(files)
        self.clone_count_label.setText(f"Samples: {count} file{'s' if count != 1 else ''}")

        self.clone_table.setRowCount(count)
        for row, (name, duration, path) in enumerate(files):
            self.clone_table.setItem(row, 0, QTableWidgetItem(name))
            self.clone_table.setItem(row, 1, QTableWidgetItem(f"{duration:.2f}"))
            self.clone_table.setItem(row, 2, QTableWidgetItem(path))

    def _on_record_clone_clip(self) -> None:
        """
        Start a background recording of a short reference clip for voice cloning.
        """
        if hasattr(self, "_clone_recorder_thread") and self._clone_recorder_thread.isRunning():
            return

        self.clone_record_button.setEnabled(False)
        self.clone_status_label.setText("Recording 5 seconds... please speak now.")

        self._clone_recorder_thread = CloneRecorderWorker(seconds=5, parent=self)
        self._clone_recorder_thread.finished.connect(self._on_clone_record_finished)
        self._clone_recorder_thread.start()

    def _on_clone_record_finished(self, path: str, success: bool, error_msg: str) -> None:
        """
        Called by CloneRecorderWorker when recording completes.
        """
        self.clone_record_button.setEnabled(True)

        if success:
            self.clone_status_label.setText(f"Saved reference clip:\n{path}")
        else:
            self.clone_status_label.setText("Recording failed.")
            QMessageBox.warning(
                self,
                "Recording failed",
                f"Could not record test clip:\n{error_msg}",
            )

    def _populate(self, table: QTableWidget, rows: List[List[str]]) -> None:
        table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, val in enumerate(row[: table.columnCount()]):
                item = QTableWidgetItem(val)
                item.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
                table.setItem(r, c, item)

    def on_record_clone_sample(self) -> None:
        """Record a short clean sample into VOICE_SAMPLES_DIR for cloning."""
        duration_sec = 5.0
        sample_rate = 16_000
        channels = 1

        if (
            QMessageBox.question(
                self,
                "Record sample",
                f"Record a new {int(duration_sec)} second sample at {sample_rate} Hz?\n\n"
                "Make sure the room is quiet and Jackson is close to the mic.",
            )
            != QMessageBox.StandardButton.Yes
        ):
            return

        try:
            self.record_btn.setEnabled(False)
            self.record_btn.setText("Recording...")
            self.record_btn.repaint()

            frames = int(duration_sec * sample_rate)
            audio = sd.rec(
                frames,
                samplerate=sample_rate,
                channels=channels,
                dtype="float32",
            )
            sd.wait()
        except Exception as exc:
            QMessageBox.critical(self, "Recording failed", f"Could not record audio:\n{exc}")
            self.record_btn.setEnabled(True)
            self.record_btn.setText("Record new sample")
            return

        os.makedirs(VOICE_SAMPLES_DIR, exist_ok=True)

        base = "jackson_sample"
        idx = 1
        while True:
            fname = f"{base}_{idx:03d}.wav"
            out_path = os.path.join(VOICE_SAMPLES_DIR, fname)
            if not os.path.exists(out_path):
                break
            idx += 1

        try:
            sf.write(out_path, audio, sample_rate)
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", f"Could not save WAV:\n{exc}")
            self.record_btn.setEnabled(True)
            self.record_btn.setText("Record new sample")
            return

        self.record_btn.setEnabled(True)
        self.record_btn.setText("Record new sample")

        QMessageBox.information(
            self,
            "Sample saved",
            f"Saved:\n{out_path}\n\nThe cloning engine will use all WAV files in this folder as reference.",
        )
        self._refresh_clone_samples()

    def on_open_clone_folder(self) -> None:
        """Open the OS file manager at the voice samples directory."""
        os.makedirs(VOICE_SAMPLES_DIR, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(VOICE_SAMPLES_DIR))

    def on_kill(self) -> None:
        if QMessageBox.question(self, "Confirm", "Send /kill to companion?") != QMessageBox.StandardButton.Yes:
            return
        post_json("/kill")

    def on_wipe(self) -> None:
        if QMessageBox.question(self, "Confirm", "Wipe logs and voice facets?") != QMessageBox.StandardButton.Yes:
            return
        post_json("/wipe")


def main() -> None:
    app = QApplication(sys.argv)
    win = CaregiverWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
