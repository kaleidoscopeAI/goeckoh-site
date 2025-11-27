"""KivyMD GUI for the Echo Crystalline Heart companion.

Child-facing view:
- Crystal avatar driven by Heart + Brain metrics.
- Last inner-voice phrase.
- Simple stress / harmony / confidence gauges.

Caregiver view:
- Text summary of last utterance, caption, Heart and Brain metrics.

Voice setup view:
- Mic check.
- Record neutral/excited/tired samples for Voice Crystal PPP.
"""

from __future__ import annotations

from typing import Optional, List
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy.io import wavfile
from kivy.clock import Clock
from kivy.metrics import dp
from kivy.uix.boxlayout import BoxLayout
from kivymd.app import MDApp
from kivymd.uix.screen import MDScreen
from kivymd.uix.screenmanager import MDScreenManager
from kivymd.uix.label import MDLabel
from kivymd.uix.button import MDRaisedButton, MDIconButton
from kivymd.uix.progressbar import MDProgressBar
from kivymd.uix.card import MDCard
from kivymd.uix.slider import MDSlider

from .system_state import SystemState, SystemSnapshot
from .controller import AvatarController
from .speech_loop import SpeechLoop
from .avatar_widget import AvatarWidget
from .config import CONFIG
from .events import HeartMetrics


class ChildScreen(MDScreen):
    """
    Jackson-facing view.
    Shows:
    - Live Crystal Avatar visualization.
    - Last inner-voice phrase in big text.
    - Brain caption ("I feel like ...").
    - Simple gauges for stress, harmony, confidence.
    """
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        root = BoxLayout(orientation="vertical", padding=dp(16), spacing=dp(12))
        
        self.avatar_widget = AvatarWidget(max_nodes=1200, size_hint=(1, 0.5))
        self.status_label = MDLabel(
            text="Inner voice is waiting...", halign="center", theme_text_color="Primary",
            font_style="H5", size_hint_y=None, height=dp(64),
        )
        self.caption_label = MDLabel(
            text="", halign="center", theme_text_color="Secondary",
            font_style="Subtitle1", size_hint_y=None, height=dp(60),
        )
        metrics_card = MDCard(
            orientation="vertical", padding=dp(12), spacing=dp(8),
            size_hint=(1, None), height=dp(180), radius=[dp(12)], elevation=3,
        )
        self.stress_bar = MDProgressBar(max=100, value=20)
        self.harmony_bar = MDProgressBar(max=100, value=80)
        self.conf_bar = MDProgressBar(max=100, value=50)

        metrics_card.add_widget(MDLabel(text="Stress", theme_text_color="Secondary", font_style="Caption"))
        metrics_card.add_widget(self.stress_bar)
        metrics_card.add_widget(MDLabel(text="Harmony", theme_text_color="Secondary", font_style="Caption"))
        metrics_card.add_widget(self.harmony_bar)
        metrics_card.add_widget(MDLabel(text="Confidence", theme_text_color="Secondary", font_style="Caption"))
        metrics_card.add_widget(self.conf_bar)

        root.add_widget(self.avatar_widget)
        root.add_widget(self.status_label)
        root.add_widget(self.caption_label)
        root.add_widget(metrics_card)
        self.add_widget(root)

    def update_from_state(self, state: SystemState) -> None:
        snap = state.get_snapshot()
        self.status_label.text = snap.last_echo.text_clean if snap.last_echo else "Inner voice is waiting..."
        self.caption_label.text = snap.caption or ""
        if snap.heart:
            self.stress_bar.value = snap.heart.stress * 100.0
            self.harmony_bar.value = snap.heart.harmony * 100.0
            self.conf_bar.value = snap.heart.confidence * 100.0
        if snap.avatar:
            self.avatar_widget.update_from_frame(snap.avatar)

class ParentScreen(MDScreen):
    """Molly-facing dashboard."""

    def __init__(self, cca_enabled: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.cca_enabled = cca_enabled
        root = BoxLayout(orientation="vertical", padding=dp(16), spacing=dp(12))
        self.title_label = MDLabel(
            text="Molly Dashboard",
            halign="center",
            font_style="H5",
            size_hint_y=None,
            height=dp(60),
        )
        summary_card = MDCard(
            orientation="vertical",
            padding=dp(12),
            spacing=dp(8),
            size_hint=(1, 1),
            radius=[dp(12)],
            elevation=3,
        )
        self.summary_label = MDLabel(
            text="No data yet.", halign="left", theme_text_color="Secondary"
        )
        self.drc_status_label = MDLabel(
            text="",
            halign="left",
            theme_text_color="Hint",
            font_style="Caption",
            size_hint_y=None,
            height=dp(20),
        )
        self.drc_counts_label = MDLabel(
            text="", halign="left", theme_text_color="Secondary", font_style="Caption"
        )
        self.drc_last_label = MDLabel(
            text="", halign="left", theme_text_color="Secondary", font_style="Caption"
        )
        summary_card.add_widget(self.summary_label)
        summary_card.add_widget(self.drc_status_label)
        summary_card.add_widget(self.drc_counts_label)
        summary_card.add_widget(self.drc_last_label)
        root.add_widget(self.title_label)
        root.add_widget(summary_card)
        self.add_widget(root)

    def update_from_state(self, state: SystemState) -> None:
        snap = state.get_snapshot()
        lines = []
        if self.cca_enabled:
            self.drc_status_label.text = "DRC / CCA: Online (policy + GCL gated)"
        else:
            self.drc_status_label.text = "DRC / CCA: Offline"
        if snap.last_echo:
            lines.append(f"Last phrase (Jackson, first-person): {snap.last_echo.text_clean}")
        if snap.caption:
            lines.append(f"Inner caption: {snap.caption}")
        if snap.heart:
            h = snap.heart
            lines.append(f"Heart: Stress={h.stress:.2f}, Harmony={h.harmony:.2f}, Energy={h.energy:.2f}, Confidence={h.confidence:.2f}, Temp={h.temperature:.2f}")
        if snap.brain:
            b = snap.brain
            lines.append(f"Brain: H_bits={b.H_bits:.3f}, S_field={b.S_field:.3f}, L={b.L:.3f}, Coherence={b.coherence:.3f}, Phi={b.phi:.3f}")
        self.summary_label.text = "\n\n".join(lines) if lines else "No data yet."

        # Lightweight DRC activity summary from guidance CSV
        try:
            import csv
            from .config import CONFIG as _CFG

            path = _CFG.paths.guidance_csv
            total_speech = 0
            total_aba = 0
            last_msg = ""
            if path.exists():
                with path.open() as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        event = (row.get("event") or "").strip()
                        if event == "cca_speech":
                            total_speech += 1
                            last_msg = row.get("message") or last_msg
                        elif event == "cca_aba":
                            total_aba += 1
                            last_msg = row.get("message") or last_msg
            if total_speech or total_aba:
                self.drc_counts_label.text = (
                    f"CCA speech: {total_speech} • CCA ABA: {total_aba}"
                )
                self.drc_last_label.text = (
                    f"Last CCA message: \"{last_msg}\"" if last_msg else ""
                )
            else:
                self.drc_counts_label.text = "No CCA activity yet."
                self.drc_last_label.text = ""
        except Exception:
            # Fail silently; keep UI responsive even if CSV is malformed.
            pass

class VoiceSetupScreen(MDScreen):
    """Voice setup wizard for Jackson."""
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        root = BoxLayout(orientation="vertical", padding=dp(16), spacing=dp(16))
        title = MDLabel(text="Voice Setup – Jackson", halign="center", font_style="H5", size_hint_y=None, height=dp(40))
        self.status_label = MDLabel(text="Follow steps to record voice facets.", halign="left", theme_text_color="Secondary")
        
        button_row = BoxLayout(orientation="horizontal", spacing=dp(8), size_hint_y=None, height=dp(48))
        button_row.add_widget(MDRaisedButton(text="Mic Check", on_release=lambda *_: self._mic_check()))
        button_row.add_widget(MDRaisedButton(text="Record Neutral", on_release=lambda *_: self._start_recording("neutral")))
        button_row.add_widget(MDRaisedButton(text="Record Excited", on_release=lambda *_: self._start_recording("excited")))
        button_row.add_widget(MDRaisedButton(text="Record Tired", on_release=lambda *_: self._start_recording("tired")))

        volume_card = MDCard(orientation="vertical", padding=dp(12), spacing=dp(8), size_hint=(1, None), height=dp(140), radius=[dp(12)], elevation=3)
        # Inner voice volume is currently baked into VoiceCrystalConfig; keep slider UI for future wiring.
        self.volume_slider = MDSlider(min=0.1, max=1.0, value=0.5, step=0.05)
        self.volume_slider.bind(value=self._on_volume_change)
        volume_card.add_widget(MDLabel(text="Inner Voice Volume", halign="left", theme_text_color="Secondary", size_hint_y=None, height=dp(24)))
        volume_card.add_widget(self.volume_slider)
        volume_card.add_widget(MDRaisedButton(text="Test Inner Voice", on_release=lambda *_: self._test_inner_voice(), size_hint=(1, None), height=dp(40)))

        root.add_widget(title)
        root.add_widget(self.status_label)
        root.add_widget(button_row)
        root.add_widget(volume_card)
        self.add_widget(root)

    def _update_status(self, text: str):
        Clock.schedule_once(lambda dt: setattr(self.status_label, 'text', text), 0)

    def _mic_check(self):
        def worker():
            try:
                self._update_status("Mic check: listening for 1 second...")
                data = sd.rec(int(1.0 * CONFIG.audio.sample_rate), samplerate=CONFIG.audio.sample_rate, channels=1, dtype="float32")
                sd.wait()
                rms = float(np.sqrt(np.mean(data ** 2)))
                self._update_status(f"Mic level RMS: {rms:.4f} (speak at a natural volume)")
            except Exception as e:
                self._update_status(f"Mic check error: {e}")
        threading.Thread(target=worker, daemon=True).start()

    def _start_recording(self, sample_type: str):
        def worker():
            try:
                self._update_status(f"Recording {sample_type} in 3… 2… 1…")
                time.sleep(1.0)
                self._update_status(f"Recording {sample_type}… speak now.")
                data = sd.rec(int(3.0 * CONFIG.audio.sample_rate), samplerate=CONFIG.audio.sample_rate, channels=1, dtype="float32")
                sd.wait()
                
                samples_dir: Path = CONFIG.paths.voices_dir
                samples_dir.mkdir(parents=True, exist_ok=True)
                existing = sorted(samples_dir.glob(f"jackson_{sample_type}_*.wav"))
                out_path = samples_dir / f"jackson_{sample_type}_{len(existing) + 1}.wav"
                
                wavfile.write(out_path, CONFIG.audio.sample_rate, (data * 32767).astype(np.int16))
                self._update_status(f"Saved {out_path.name}")
            except Exception as e:
                self._update_status(f"Recording error ({sample_type}): {e}")
        threading.Thread(target=worker, daemon=True).start()

    def _on_volume_change(self, instance, value):
        # Placeholder: plumb through to VoiceCrystalConfig if you want live control.
        pass

    def _test_inner_voice(self):
        self._update_status("Inner voice test is routed through the main loop; say a phrase on the child view.")

class EchoGuiApp(MDApp):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.state: Optional[SystemState] = None
        self.avatar: Optional[AvatarController] = None
        self.speech: Optional[SpeechLoop] = None
        self.sm: Optional[MDScreenManager] = None
        self.child_screen: Optional[ChildScreen] = None
        self.parent_screen: Optional[ParentScreen] = None
        self.voice_screen: Optional[VoiceSetupScreen] = None

    def build(self):
        self.title = "Echo – Crystalline Heart Companion"
        self.state = SystemState()
        self.avatar = AvatarController()
        self.speech = SpeechLoop(config=CONFIG, state=self.state, avatar=self.avatar)
        
        root = BoxLayout(orientation="vertical")
        button_bar = BoxLayout(orientation="horizontal", size_hint_y=None, height=dp(56), padding=dp(8), spacing=dp(8))
        btn_child = MDRaisedButton(text="Jackson View", on_release=lambda *_: self.switch_to("child"))
        btn_parent = MDRaisedButton(text="Molly Dashboard", on_release=lambda *_: self.switch_to("parent"))
        btn_voice = MDRaisedButton(text="Voice Setup", on_release=lambda *_: self.switch_to("voice"))
        button_bar.add_widget(btn_child)
        button_bar.add_widget(btn_parent)
        button_bar.add_widget(btn_voice)

        self.sm = MDScreenManager()
        self.child_screen = ChildScreen(name="child")
        cca_enabled = bool(self.speech and self.speech.cca_bridge)
        self.parent_screen = ParentScreen(name="parent", cca_enabled=cca_enabled)
        self.voice_screen = VoiceSetupScreen(name="voice")
        self.sm.add_widget(self.child_screen)
        self.sm.add_widget(self.parent_screen)
        self.sm.add_widget(self.voice_screen)
        
        root.add_widget(button_bar)
        root.add_widget(self.sm)
        return root

    def switch_to(self, name: str):
        if not self.sm or not self.speech: return
        if name == "voice":
            self.speech.stop()
        else:
            self.speech.start()
        self.sm.current = name

    def on_start(self):
        self.switch_to("voice") # Start in voice setup so mic is free
        Clock.schedule_interval(self._refresh_ui, 0.5)

    def _refresh_ui(self, dt: float):
        if not self.state: return
        if self.child_screen and self.sm and self.sm.current == 'child':
            self.child_screen.update_from_state(self.state)
        if self.parent_screen and self.sm and self.sm.current == 'parent':
            self.parent_screen.update_from_state(self.state)

    def on_stop(self):
        if self.speech:
            self.speech.stop()
