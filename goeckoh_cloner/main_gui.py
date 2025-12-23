# main_gui.py
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import webbrowser
from pathlib import Path

from hardware_audit import get_hardware_summary
from licensing_security import (
    check_license_valid,
    activate_license,
    load_license,
)
from voice_logger import enroll_child
from goeckoh_loop import GoeckohLoop

# --- BRANDING ---
ASSETS_DIR = Path(__file__).resolve().parent / "assets"
LOGO_IMG_PATH = ASSETS_DIR / "goeckoh.png"   # export the logo you sent as this file


class GoeckohApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Goeckoh Neuro-Acoustic Cloner")
        self.geometry("700x500")

        self.loop_thread = None
        self.loop_stop_event = threading.Event()

        self.logo_img = None

        self._build_ui()
        self._refresh_license_status()
        self._populate_hardware_info()

    def _build_ui(self):
        # Top branding strip
        header = ttk.Frame(self)
        header.pack(fill="x", pady=(4, 0), padx=6)

        # Logo on the left (if present)
        left = ttk.Frame(header)
        left.pack(side="left", anchor="w")

        if LOGO_IMG_PATH.exists():
            try:
                # Use a small resized PNG of the logo
                img = tk.PhotoImage(file=str(LOGO_IMG_PATH))
                # Optionally subsample if it's large
                if img.width() > 160:
                    factor = max(1, int(img.width() / 160))
                    img = img.subsample(factor, factor)
                self.logo_img = img
                logo_lbl = ttk.Label(left, image=self.logo_img)
                logo_lbl.pack(side="left", padx=(0, 8))
            except Exception:
                pass

        title_block = ttk.Frame(left)
        title_block.pack(side="left")

        title_lbl = ttk.Label(
            title_block,
            text="GOECKOH",
            font=("TkDefaultFont", 13, "bold")
        )
        title_lbl.pack(anchor="w")

        subtitle_lbl = ttk.Label(
            title_block,
            text="Neuro-Acoustic Cloner • Inner Speech Mirror",
            font=("TkDefaultFont", 9),
            foreground="#64748b",
        )
        subtitle_lbl.pack(anchor="w")

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, pady=(6, 0))

        # Tabs
        self.setup_frame = ttk.Frame(notebook)
        self.enroll_frame = ttk.Frame(notebook)
        self.run_frame = ttk.Frame(notebook)

        notebook.add(self.setup_frame, text="Setup")
        notebook.add(self.enroll_frame, text="Voice Crystal Seed")
        notebook.add(self.run_frame, text="Echo Loop")

        self._build_setup_tab()
        self._build_enroll_tab()
        self._build_run_tab()

    # ---------- SETUP TAB ----------
    def _build_setup_tab(self):
        f = self.setup_frame

        # Hardware
        hw_label = ttk.Label(f, text="Hardware Summary", font=("TkDefaultFont", 11, "bold"))
        hw_label.pack(anchor="w", padx=10, pady=(10, 4))

        self.hw_text = tk.Text(f, height=8, width=80, state="disabled", bg="#0f172a", fg="#e5e7eb")
        self.hw_text.pack(fill="x", padx=10)

        # License
        lic_label = ttk.Label(f, text="License Activation", font=("TkDefaultFont", 11, "bold"))
        lic_label.pack(anchor="w", padx=10, pady=(10, 4))

        lic_frame = ttk.Frame(f)
        lic_frame.pack(fill="x", padx=10)

        ttk.Label(lic_frame, text="License Key:").grid(row=0, column=0, sticky="w")
        self.lic_entry = ttk.Entry(lic_frame, width=30)
        self.lic_entry.grid(row=0, column=1, padx=5)

        self.lic_status_label = ttk.Label(lic_frame, text="Status: Unknown")
        self.lic_status_label.grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))

        activate_btn = ttk.Button(lic_frame, text="Activate", command=self._on_activate_license)
        activate_btn.grid(row=0, column=2, padx=5)

    def _populate_hardware_info(self):
        summary = get_hardware_summary()
        lines = [
            f"Operating System: {summary.os}",
            f"CPU: {summary.cpu}",
            f"Cores: {summary.cores}",
            f"CUDA GPU: {'Yes' if summary.has_cuda else 'No'}",
            f"CUDA Device: {summary.cuda_device}",
            f"Suggested Whisper model: {summary.suggested_whisper_model}",
        ]
        self.hw_text.configure(state="normal")
        self.hw_text.delete("1.0", tk.END)
        self.hw_text.insert(tk.END, "\n".join(lines))
        self.hw_text.configure(state="disabled")

    def _refresh_license_status(self):
        if check_license_valid():
            lic = load_license()
            key_display = lic.key if lic else "(loaded)"
            self.lic_status_label.config(
                text=f"Status: Activated ({key_display})"
            )
        else:
            self.lic_status_label.config(
                text="Status: Not Activated"
            )

    def _on_activate_license(self):
        key = self.lic_entry.get().strip()
        if not key:
            messagebox.showerror("Error", "Please enter a license key.")
            return
        ok = activate_license(key)
        if not ok:
            messagebox.showerror("Error", "Invalid license format. Use e.g. GK-ABCD-1234.")
            return
        self._refresh_license_status()
        messagebox.showinfo("Success", "License activated on this machine.")

    # ---------- ENROLL TAB ----------
    def _build_enroll_tab(self):
        f = self.enroll_frame

        ttk.Label(f, text="Enroll Child Voice", font=("TkDefaultFont", 11, "bold")).pack(
            anchor="w", padx=10, pady=(10, 4)
        )

        form = ttk.Frame(f)
        form.pack(padx=10, pady=10, anchor="w")

        ttk.Label(form, text="Profile Name (e.g. jackson):").grid(row=0, column=0, sticky="w")
        self.enroll_name_entry = ttk.Entry(form, width=25)
        self.enroll_name_entry.grid(row=0, column=1, padx=5)

        ttk.Label(form, text="Record Duration (sec):").grid(row=1, column=0, sticky="w", pady=(5, 0))
        self.enroll_dur_entry = ttk.Entry(form, width=10)
        self.enroll_dur_entry.insert(0, "10")
        self.enroll_dur_entry.grid(row=1, column=1, sticky="w", padx=5, pady=(5, 0))

        enroll_btn = ttk.Button(form, text="Start Enrollment", command=self._on_enroll)
        enroll_btn.grid(row=2, column=0, columnspan=2, pady=(10, 0))

        ttk.Label(
            f,
            text=(
                "During enrollment, let the child vocalize freely.\n"
                "Goeckoh will build their Bubble DNA (VoiceFingerprint) and register\n"
                "a short Voice Crystal seed clip with the cloning engine."
            ),
            wraplength=650,
            justify="left",
        ).pack(padx=10, pady=10, anchor="w")

    def _on_enroll(self):
        name = self.enroll_name_entry.get().strip()
        if not name:
            messagebox.showerror("Error", "Please enter a profile name.")
            return
        try:
            duration = float(self.enroll_dur_entry.get().strip())
        except ValueError:
            messagebox.showerror("Error", "Duration must be a number.")
            return

        if duration < 3.0:
            if not messagebox.askyesno(
                "Confirm", "Duration is short. Enrollment may be weak. Continue?"
            ):
                return

        def worker():
            try:
                enroll_child(name, duration_sec=duration)
                messagebox.showinfo("Success", f"Enrollment complete for profile '{name}'.")
            except Exception as e:
                messagebox.showerror("Error", f"Enrollment failed:\n{e}")

        threading.Thread(target=worker, daemon=True).start()

    # ---------- RUN TAB ----------
    def _build_run_tab(self):
        f = self.run_frame

        ttk.Label(f, text="Run Goeckoh Loop", font=("TkDefaultFont", 11, "bold")).pack(
            anchor="w", padx=10, pady=(10, 4)
        )

        form = ttk.Frame(f)
        form.pack(padx=10, pady=10, anchor="w")

        ttk.Label(form, text="Profile Name:").grid(row=0, column=0, sticky="w")
        self.run_name_entry = ttk.Entry(form, width=25)
        self.run_name_entry.grid(row=0, column=1, padx=5)

        start_btn = ttk.Button(form, text="Start Loop", command=self._on_start_loop)
        start_btn.grid(row=1, column=0, pady=(10, 0))

        stop_btn = ttk.Button(form, text="Stop Loop", command=self._on_stop_loop)
        stop_btn.grid(row=1, column=1, pady=(10, 0), padx=5)

        open_ui_btn = ttk.Button(form, text="Open Bubble Viewer", command=self._on_open_bubble_viewer)
        open_ui_btn.grid(row=2, column=0, columnspan=2, pady=(10, 0))

        ttk.Label(
            f,
            text="When the loop is running, the system will listen for speech,\n"
                 "correct it to first-person, clone it in the child's voice, and animate the Voice Bubble.",
            wraplength=650,
            justify="left",
        ).pack(padx=10, pady=10, anchor="w")

    def _on_start_loop(self):
        if not check_license_valid():
            messagebox.showerror("Error", "No valid license. Activate on the Setup tab.")
            return
        if self.loop_thread and self.loop_thread.is_alive():
            messagebox.showwarning("Running", "Loop is already running.")
            return
        name = self.run_name_entry.get().strip()
        if not name:
            messagebox.showerror("Error", "Please enter a profile name.")
            return

        self.loop_stop_event.clear()

        def worker():
            try:
                loop = GoeckohLoop(name, stop_event=self.loop_stop_event)
                loop.start()
            except Exception as e:
                messagebox.showerror("Error", f"Goeckoh loop failed:\n{e}")

        self.loop_thread = threading.Thread(target=worker, daemon=True)
        self.loop_thread.start()

    def _on_stop_loop(self):
        if self.loop_thread and self.loop_thread.is_alive():
            self.loop_stop_event.set()
            messagebox.showinfo("Stopping", "Requested loop stop. It will exit after current utterance.")
        else:
            messagebox.showinfo("Not running", "Loop is not currently running.")

    def _on_open_bubble_viewer(self):
        webbrowser.open_new_tab("webui/bubble_viewer.html")


if __name__ == "__main__":
    app = GoeckohApp()
    app.mainloop()
