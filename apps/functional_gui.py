#!/usr/bin/env python3
"""
Goeckoh Functional GUI - Fully Integrated with Real System
=========================================================
Professional GUI that actually connects to and controls the real Goeckoh system:
- Real CrystallineHeart integration
- Real AudioBridge integration  
- Real-time voice pipeline visualization
- Functional controls that actually work
"""

import sys
import numpy as np
from pathlib import Path
from collections import deque
import time
import math
import threading

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTextEdit, QLineEdit, QSplitter, QFrame,
    QGridLayout, QGroupBox, QProgressBar, QSlider
)
from PySide6.QtCore import Qt, QTimer, QPointF, Signal, QThread, QSize
from PySide6.QtGui import (
    QFont, QColor, QPainter, QPen, QBrush, QLinearGradient,
    QRadialGradient, QPolygonF, QPainterPath
)

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import REAL system components
HEART_AVAILABLE = False
AUDIO_AVAILABLE = False

# Try multiple import paths for CrystallineHeart
for import_path in [
    'goeckoh.heart.logic_core',
    'heart.logic_core',
    'go_eckoh.heart.logic_core'
]:
    try:
        module = __import__(import_path, fromlist=['CrystallineHeart'])
        CrystallineHeart = getattr(module, 'CrystallineHeart')
        HEART_AVAILABLE = True
        break
    except (ImportError, AttributeError):
        continue

if not HEART_AVAILABLE:
    print("Warning: CrystallineHeart not available")

# Try multiple import paths for AudioBridge
for import_path in [
    'goeckoh.audio.audio_bridge',
    'audio.audio_bridge',
    'go_eckoh.audio.audio_bridge'
]:
    try:
        module = __import__(import_path, fromlist=['AudioBridge'])
        AudioBridge = getattr(module, 'AudioBridge')
        AUDIO_AVAILABLE = True
        break
    except (ImportError, AttributeError):
        continue

if not AUDIO_AVAILABLE:
    print("Warning: AudioBridge not available")


class HeartVisualizationWidget(QWidget):
    """Real-time visualization of the Crystalline Heart lattice"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 400)
        
        # Data from real heart
        self.nodes = np.zeros(1024, dtype=np.float32)
        self.gcl = 1.0
        self.stress = 0.0
        self.mode_label = "INIT"
        self.color = (0.0, 1.0, 1.0, 1.0)
        
        # Animation
        self.animation_phase = 0.0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(50)  # 20 FPS
        
    def update_animation(self):
        self.animation_phase += 0.03
        if self.animation_phase > 2 * math.pi:
            self.animation_phase = 0.0
        self.update()
    
    def update_from_heart(self, nodes, gcl, stress, mode_label, color):
        """Update visualization from real heart data"""
        if len(nodes) == 1024:
            self.nodes = np.array(nodes, dtype=np.float32)
        self.gcl = gcl
        self.stress = stress
        self.mode_label = mode_label
        self.color = color
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        width = self.width()
        height = self.height()
        center_x = width // 2
        center_y = height // 2
        
        # Dark gradient background
        gradient = QRadialGradient(center_x, center_y, width)
        gradient.setColorAt(0, QColor(15, 23, 42))
        gradient.setColorAt(1, QColor(2, 6, 23))
        painter.fillRect(self.rect(), QBrush(gradient))
        
        # Draw lattice structure
        size = int(np.sqrt(len(self.nodes)))  # 32x32 grid
        node_size = min(width, height) / (size * 1.5)
        
        max_node = max(abs(n) for n in self.nodes) if len(self.nodes) > 0 else 1.0
        if max_node == 0:
            max_node = 1.0
        
        # Draw connections
        pen = QPen(QColor(99, 102, 241, 40), 1)
        painter.setPen(pen)
        
        for i in range(len(self.nodes) - 1):
            if i % size < size - 1:  # Right neighbor
                x1 = (i % size) * node_size * 1.5 + center_x - (size * node_size * 1.5) / 2
                y1 = (i // size) * node_size * 1.5 + center_y - (size * node_size * 1.5) / 2
                x2 = ((i+1) % size) * node_size * 1.5 + center_x - (size * node_size * 1.5) / 2
                y2 = ((i+1) // size) * node_size * 1.5 + center_y - (size * node_size * 1.5) / 2
                
                intensity = (abs(self.nodes[i]) + abs(self.nodes[i+1])) / (2.0 * max_node)
                pen.setColor(QColor(99, 102, 241, int(40 * intensity)))
                painter.setPen(pen)
                painter.drawLine(int(x1), int(y1), int(x2), int(y2))
        
        # Draw nodes
        for i, node_val in enumerate(self.nodes):
            x = (i % size) * node_size * 1.5 + center_x - (size * node_size * 1.5) / 2
            y = (i // size) * node_size * 1.5 + center_y - (size * node_size * 1.5) / 2
            
            # Node size based on value
            node_radius = 2 + abs(node_val) / max_node * 4
            
            # Color based on value and GCL
            r, g, b, a = self.color
            alpha = int(150 + 105 * abs(node_val) / max_node)
            color = QColor(int(r * 255), int(g * 255), int(b * 255), alpha)
            
            # Pulsing effect
            pulse = 1.0 + 0.15 * math.sin(self.animation_phase + i * 0.01)
            node_radius *= pulse
            
            # Draw node with glow
            painter.setPen(Qt.NoPen)
            radial_grad = QRadialGradient(x, y, node_radius * 3)
            radial_grad.setColorAt(0, color)
            radial_grad.setColorAt(1, QColor(color.red(), color.green(), color.blue(), 0))
            painter.setBrush(QBrush(radial_grad))
            painter.drawEllipse(QPointF(x, y), node_radius * 3, node_radius * 3)
            
            # Draw solid node
            painter.setBrush(QBrush(color))
            painter.drawEllipse(QPointF(x, y), node_radius, node_radius)
        
        # Draw GCL indicator
        gcl_text = f"GCL: {self.gcl:.3f} | {self.mode_label}"
        painter.setPen(QPen(QColor(255, 255, 255)))
        font = QFont("Inter", 16, QFont.Bold)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        text_rect = metrics.boundingRect(gcl_text)
        painter.drawText(center_x - text_rect.width() // 2, 30, gcl_text)


class MetricsWidget(QWidget):
    """Real-time metrics display"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(150)
        
        layout = QGridLayout(self)
        layout.setSpacing(15)
        
        # GCL Card
        self.gcl_label = QLabel("GCL: 1.000")
        self.gcl_label.setStyleSheet("""
            QLabel {
                background-color: rgba(15, 23, 42, 0.8);
                border: 2px solid rgba(99, 102, 241, 0.5);
                border-radius: 12px;
                padding: 20px;
                color: #6366f1;
                font-size: 24px;
                font-weight: bold;
            }
        """)
        layout.addWidget(self.gcl_label, 0, 0)
        
        # Stress Card
        self.stress_label = QLabel("Stress: 0.000")
        self.stress_label.setStyleSheet("""
            QLabel {
                background-color: rgba(15, 23, 42, 0.8);
                border: 2px solid rgba(236, 72, 153, 0.5);
                border-radius: 12px;
                padding: 20px;
                color: #ec4899;
                font-size: 24px;
                font-weight: bold;
            }
        """)
        layout.addWidget(self.stress_label, 0, 1)
        
        # Mode Card
        self.mode_label = QLabel("Mode: INIT")
        self.mode_label.setStyleSheet("""
            QLabel {
                background-color: rgba(15, 23, 42, 0.8);
                border: 2px solid rgba(139, 92, 246, 0.5);
                border-radius: 12px;
                padding: 20px;
                color: #8b5cf6;
                font-size: 20px;
                font-weight: bold;
            }
        """)
        layout.addWidget(self.mode_label, 1, 0, 1, 2)
    
    def update_metrics(self, gcl, stress, mode):
        """Update metrics from real data"""
        self.gcl_label.setText(f"GCL: {gcl:.3f}")
        self.stress_label.setText(f"Stress: {stress:.3f}")
        self.mode_label.setText(f"Mode: {mode}")


class FunctionalGoeckohGUI(QMainWindow):
    """Fully functional GUI integrated with real Goeckoh system"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Goeckoh Neuro-Acoustic Exocortex - Functional Interface")
        self.setGeometry(50, 50, 1600, 1000)
        
        # Initialize REAL system components
        self.heart = None
        self.audio = None
        
        if HEART_AVAILABLE:
            try:
                self.heart = CrystallineHeart()
                print("[GUI] CrystallineHeart initialized")
            except Exception as e:
                print(f"[GUI] Error initializing heart: {e}")
        
        if AUDIO_AVAILABLE:
            try:
                self.audio = AudioBridge()
                print("[GUI] AudioBridge initialized")
            except Exception as e:
                print(f"[GUI] Error initializing audio: {e}")
        
        # Setup UI
        self.setup_ui()
        self.setup_styling()
        
        # Update timer - connects to REAL system
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_from_system)
        self.update_timer.start(100)  # 10 Hz - matches heart update rate
        
        # System state
        self.system_running = False
        
    def setup_ui(self):
        """Setup the functional UI"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Top bar
        top_bar = self.create_top_bar()
        main_layout.addWidget(top_bar)
        
        # Main content splitter
        main_splitter = QSplitter(Qt.Horizontal)
        
        # Left: Heart visualization
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        heart_group = QGroupBox("Crystalline Heart Lattice (1024 Nodes)")
        heart_group.setStyleSheet(self.get_groupbox_style())
        heart_layout = QVBoxLayout(heart_group)
        
        self.heart_widget = HeartVisualizationWidget()
        heart_layout.addWidget(self.heart_widget)
        
        left_layout.addWidget(heart_group)
        main_splitter.addWidget(left_panel)
        
        # Right: Controls and metrics
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(15)
        
        # Metrics
        self.metrics_widget = MetricsWidget()
        right_layout.addWidget(self.metrics_widget)
        
        # Response display
        response_group = QGroupBox("System Response")
        response_group.setStyleSheet(self.get_groupbox_style())
        response_layout = QVBoxLayout(response_group)
        
        self.response_text = QTextEdit()
        self.response_text.setReadOnly(True)
        self.response_text.setMinimumHeight(200)
        self.response_text.setStyleSheet("""
            QTextEdit {
                background-color: rgba(30, 41, 59, 0.8);
                border: 1px solid rgba(99, 102, 241, 0.3);
                border-radius: 8px;
                padding: 15px;
                color: #e2e8f0;
                font-size: 14px;
            }
        """)
        response_layout.addWidget(self.response_text)
        right_layout.addWidget(response_group)
        
        # Input area
        input_group = QGroupBox("Input")
        input_group.setStyleSheet(self.get_groupbox_style())
        input_layout = QVBoxLayout(input_group)
        
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Enter text to process...")
        self.input_field.setStyleSheet("""
            QLineEdit {
                background-color: rgba(30, 41, 59, 0.8);
                border: 2px solid rgba(99, 102, 241, 0.3);
                border-radius: 8px;
                padding: 12px;
                color: #e2e8f0;
                font-size: 14px;
            }
            QLineEdit:focus {
                border: 2px solid #6366f1;
            }
        """)
        self.input_field.returnPressed.connect(self.process_input)
        input_layout.addWidget(self.input_field)
        
        send_btn = QPushButton("Process")
        send_btn.setStyleSheet(self.get_button_style("primary"))
        send_btn.clicked.connect(self.process_input)
        input_layout.addWidget(send_btn)
        
        right_layout.addWidget(input_group)
        
        right_layout.addStretch()
        main_splitter.addWidget(right_panel)
        
        main_splitter.setStretchFactor(0, 2)
        main_splitter.setStretchFactor(1, 1)
        
        main_layout.addWidget(main_splitter)
        
        # Status bar
        status_bar = self.create_status_bar()
        main_layout.addWidget(status_bar)
    
    def create_top_bar(self):
        """Create top control bar"""
        frame = QFrame()
        frame.setStyleSheet("""
            QFrame {
                background-color: rgba(15, 23, 42, 0.9);
                border: 1px solid rgba(99, 102, 241, 0.3);
                border-radius: 12px;
                padding: 15px;
            }
        """)
        
        layout = QHBoxLayout(frame)
        layout.setSpacing(20)
        
        # Title
        title = QLabel("Goeckoh Neuro-Acoustic Exocortex")
        title_font = QFont("Inter", 22, QFont.Bold)
        title.setFont(title_font)
        title.setStyleSheet("color: #6366f1;")
        layout.addWidget(title)
        
        layout.addStretch()
        
        # Control buttons
        self.start_btn = QPushButton("▶ Start System")
        self.start_btn.setStyleSheet(self.get_button_style("primary"))
        self.start_btn.clicked.connect(self.start_system)
        layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("⏹ Stop System")
        self.stop_btn.setStyleSheet(self.get_button_style("danger"))
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_system)
        layout.addWidget(self.stop_btn)
        
        return frame
    
    def create_status_bar(self):
        """Create status bar"""
        frame = QFrame()
        frame.setStyleSheet("""
            QFrame {
                background-color: rgba(15, 23, 42, 0.9);
                border: 1px solid rgba(99, 102, 241, 0.3);
                border-radius: 8px;
                padding: 10px;
            }
        """)
        
        layout = QHBoxLayout(frame)
        
        heart_status = "✓" if self.heart else "✗"
        audio_status = "✓" if self.audio else "✗"
        
        self.status_label = QLabel(f"● System: Ready | Heart: {heart_status} | Audio: {audio_status}")
        self.status_label.setStyleSheet("color: #10b981; font-weight: 600;")
        layout.addWidget(self.status_label)
        
        return frame
    
    def setup_styling(self):
        """Setup application-wide styling"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #020617;
            }
            QWidget {
                background-color: transparent;
                color: #e2e8f0;
                font-family: 'Inter', 'Segoe UI', sans-serif;
            }
        """)
    
    def get_button_style(self, style_type: str):
        """Get button style"""
        styles = {
            "primary": """
                QPushButton {
                    background-color: #6366f1;
                    color: white;
                    border: none;
                    padding: 12px 24px;
                    border-radius: 8px;
                    font-weight: 600;
                    font-size: 14px;
                }
                QPushButton:hover {
                    background-color: #4f46e5;
                }
                QPushButton:pressed {
                    background-color: #4338ca;
                }
                QPushButton:disabled {
                    background-color: #334155;
                    color: #64748b;
                }
            """,
            "danger": """
                QPushButton {
                    background-color: #ef4444;
                    color: white;
                    border: none;
                    padding: 12px 24px;
                    border-radius: 8px;
                    font-weight: 600;
                    font-size: 14px;
                }
                QPushButton:hover {
                    background-color: #dc2626;
                }
                QPushButton:pressed {
                    background-color: #b91c1c;
                }
            """
        }
        return styles.get(style_type, styles["primary"])
    
    def get_groupbox_style(self):
        """Get group box style"""
        return """
            QGroupBox {
                background-color: rgba(15, 23, 42, 0.6);
                border: 1px solid rgba(99, 102, 241, 0.3);
                border-radius: 12px;
                padding: 15px;
                margin-top: 10px;
                font-weight: 600;
                color: #e2e8f0;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 5px;
                color: #6366f1;
            }
        """
    
    def start_system(self):
        """Start the system"""
        if self.system_running:
            return
        
        self.system_running = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("● System: Running")
        self.status_label.setStyleSheet("color: #10b981; font-weight: 600;")
        self.response_text.append("=" * 60)
        self.response_text.append("System Started")
        self.response_text.append("Crystalline Heart: Active")
        if self.audio:
            self.response_text.append("Audio Bridge: Active")
        self.response_text.append("=" * 60)
    
    def stop_system(self):
        """Stop the system"""
        if not self.system_running:
            return
        
        self.system_running = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("● System: Stopped")
        self.status_label.setStyleSheet("color: #ef4444; font-weight: 600;")
        self.response_text.append("=" * 60)
        self.response_text.append("System Stopped")
        self.response_text.append("=" * 60)
        
        if self.audio:
            self.audio.running = False
    
    def process_input(self):
        """Process user input with REAL system"""
        text = self.input_field.text().strip()
        if not text:
            return
        
        self.input_field.clear()
        
        if not self.system_running:
            self.response_text.append("⚠ System not started. Click 'Start System' first.")
            return
        
        if not self.heart:
            self.response_text.append("⚠ CrystallineHeart not available")
            return
        
        # Process with REAL heart
        try:
            response, metrics = self.heart.process_input(text)
            
            # Update UI
            self.response_text.append(f"> Input: {text}")
            self.response_text.append(f"< Response: {response}")
            self.response_text.append("")
            
            # Trigger audio if available
            if self.audio:
                arousal = 1.0 - metrics.gcl
                self.audio.trigger(response, arousal)
            
        except Exception as e:
            self.response_text.append(f"Error: {str(e)}")
    
    def update_from_system(self):
        """Update UI from REAL system data"""
        if not self.heart or not self.system_running:
            return
        
        try:
            # Get idle update from heart (empty string = idle)
            response, metrics = self.heart.process_input("")
            
            # Update heart visualization
            self.heart_widget.update_from_heart(
                self.heart.nodes,
                metrics.gcl,
                metrics.stress,
                metrics.mode_label,
                metrics.gui_color
            )
            
            # Update metrics display
            self.metrics_widget.update_metrics(
                metrics.gcl,
                metrics.stress,
                metrics.mode_label
            )
            
        except Exception as e:
            print(f"Update error: {e}")


def main():
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Goeckoh Functional GUI")
    app.setOrganizationName("Goeckoh")
    
    # Set high DPI scaling
    app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    window = FunctionalGoeckohGUI()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

