#!/usr/bin/env python3
"""
Goeckoh Advanced GUI - Groundbreaking Therapeutic Interface
============================================================
Professional-grade GUI with real-time visualizations, advanced metrics,
and cutting-edge design matching industry-leading therapeutic software.
"""

import sys
import numpy as np
from pathlib import Path
from typing import Optional
from collections import deque
import time
import math

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTextEdit, QLineEdit, QSplitter, QFrame,
    QGridLayout, QScrollArea, QGroupBox, QSlider, QProgressBar
)
from PySide6.QtCore import Qt, QTimer, QPointF, QRectF, Signal, QThread, QSize
from PySide6.QtGui import (
    QFont, QPalette, QColor, QPainter, QPen, QBrush, QLinearGradient,
    QRadialGradient, QPolygonF, QPainterPath, QFontMetrics
)
from PySide6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis, QDateTimeAxis

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class VoiceWaveformWidget(QWidget):
    """Real-time voice waveform visualization with advanced rendering"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(200)
        self.setMinimumWidth(400)
        
        # Data buffers
        self.audio_data = deque(maxlen=2000)
        self.spectrum_data = deque(maxlen=512)
        self.time_data = deque(maxlen=2000)
        
        # Visualization settings
        self.amplitude_scale = 1.0
        self.show_spectrum = True
        self.show_waveform = True
        
        # Animation
        self.animation_phase = 0.0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(16)  # ~60 FPS
        
    def update_animation(self):
        self.animation_phase += 0.05
        if self.animation_phase > 2 * math.pi:
            self.animation_phase = 0.0
        self.update()
    
    def add_audio_sample(self, sample: float, timestamp: float = None):
        """Add audio sample for visualization"""
        self.audio_data.append(sample)
        if timestamp:
            self.time_data.append(timestamp)
        else:
            self.time_data.append(time.time())
        self.update()
    
    def add_spectrum(self, spectrum: np.ndarray):
        """Add frequency spectrum data"""
        self.spectrum_data.extend(spectrum[:512])
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        width = self.width()
        height = self.height()
        
        # Dark gradient background
        gradient = QLinearGradient(0, 0, 0, height)
        gradient.setColorAt(0, QColor(15, 23, 42))
        gradient.setColorAt(1, QColor(2, 6, 23))
        painter.fillRect(self.rect(), QBrush(gradient))
        
        if not self.audio_data:
            # Draw idle animation
            center_x = width // 2
            center_y = height // 2
            radius = min(width, height) // 4
            
            # Pulsing circle
            pulse_radius = radius * (1.0 + 0.2 * math.sin(self.animation_phase))
            pen = QPen(QColor(99, 102, 241, 100), 2)
            painter.setPen(pen)
            painter.setBrush(QBrush(QColor(99, 102, 241, 30)))
            painter.drawEllipse(QPointF(center_x, center_y), pulse_radius, pulse_radius)
            return
        
        # Draw waveform
        if self.show_waveform and len(self.audio_data) > 1:
            pen = QPen(QColor(99, 102, 241), 2)
            painter.setPen(pen)
            
            path = QPainterPath()
            data_points = list(self.audio_data)
            num_points = len(data_points)
            
            if num_points > 1:
                x_step = width / (num_points - 1)
                center_y = height // 2
                amplitude = height * 0.4 * self.amplitude_scale
                
                # Normalize data
                max_val = max(abs(d) for d in data_points) if data_points else 1.0
                if max_val > 0:
                    normalized = [d / max_val for d in data_points]
                else:
                    normalized = [0.0] * num_points
                
                # Create smooth path
                path.moveTo(0, center_y)
                for i, val in enumerate(normalized):
                    x = i * x_step
                    y = center_y - (val * amplitude)
                    path.lineTo(x, y)
                
                # Draw with gradient
                gradient_pen = QLinearGradient(0, center_y - amplitude, 0, center_y + amplitude)
                gradient_pen.setColorAt(0, QColor(139, 92, 246))
                gradient_pen.setColorAt(0.5, QColor(99, 102, 241))
                gradient_pen.setColorAt(1, QColor(236, 72, 153))
                pen.setBrush(QBrush(gradient_pen))
                painter.setPen(pen)
                painter.drawPath(path)
                
                # Fill area under curve
                fill_path = QPainterPath(path)
                fill_path.lineTo(width, center_y)
                fill_path.lineTo(0, center_y)
                fill_path.closeSubpath()
                
                fill_gradient = QLinearGradient(0, 0, 0, height)
                fill_gradient.setColorAt(0, QColor(99, 102, 241, 80))
                fill_gradient.setColorAt(1, QColor(99, 102, 241, 0))
                painter.fillPath(fill_path, QBrush(fill_gradient))
        
        # Draw frequency spectrum
        if self.show_spectrum and len(self.spectrum_data) > 1:
            spectrum_points = list(self.spectrum_data)
            num_bins = len(spectrum_points)
            
            if num_bins > 1:
                bar_width = width / num_bins
                max_spectrum = max(spectrum_points) if spectrum_points else 1.0
                
                for i, magnitude in enumerate(spectrum_points):
                    if max_spectrum > 0:
                        normalized = magnitude / max_spectrum
                    else:
                        normalized = 0.0
                    
                    bar_height = height * normalized * 0.6
                    x = i * bar_width
                    y = height - bar_height
                    
                    # Color based on frequency
                    hue = int(240 - (i / num_bins) * 120)  # Blue to pink
                    color = QColor.fromHsv(hue, 200, 255, 150)
                    
                    painter.fillRect(int(x), int(y), int(bar_width), int(bar_height), 
                                   QBrush(color))


class CrystallineHeartWidget(QWidget):
    """3D-like visualization of the 1024-node crystalline heart lattice"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 400)
        
        # Lattice state (1024 nodes)
        self.nodes = np.zeros(1024, dtype=np.float32)
        self.gcl = 1.0
        self.stress = 0.0
        
        # Visualization parameters
        self.node_positions = self._generate_lattice_positions()
        self.animation_phase = 0.0
        
        # Update timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(50)  # 20 FPS
        
    def _generate_lattice_positions(self):
        """Generate 3D-like positions for 1024 nodes"""
        positions = []
        size = int(np.sqrt(1024))  # 32x32 grid
        
        for i in range(1024):
            x = (i % size) / size
            y = (i // size) / size
            # Add some 3D depth variation
            z = np.sin(x * np.pi * 2) * np.cos(y * np.pi * 2) * 0.3
            positions.append((x, y, z))
        
        return np.array(positions)
    
    def update_animation(self):
        self.animation_phase += 0.02
        if self.animation_phase > 2 * math.pi:
            self.animation_phase = 0.0
        self.update()
    
    def update_lattice(self, nodes: np.ndarray, gcl: float, stress: float):
        """Update lattice state"""
        if len(nodes) == 1024:
            self.nodes = nodes.copy()
        self.gcl = gcl
        self.stress = stress
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        width = self.width()
        height = self.height()
        
        # Dark background with subtle gradient
        gradient = QRadialGradient(width/2, height/2, width)
        gradient.setColorAt(0, QColor(15, 23, 42))
        gradient.setColorAt(1, QColor(2, 6, 23))
        painter.fillRect(self.rect(), QBrush(gradient))
        
        # Draw connections between nodes
        if len(self.nodes) > 0:
            # Draw edges (connections)
            pen = QPen(QColor(99, 102, 241, 30), 1)
            painter.setPen(pen)
            
            # Connect nearby nodes
            size = int(np.sqrt(len(self.nodes)))
            for i in range(len(self.nodes) - 1):
                if i % size < size - 1:  # Right neighbor
                    x1 = self.node_positions[i][0] * width
                    y1 = self.node_positions[i][1] * height
                    x2 = self.node_positions[i+1][0] * width
                    y2 = self.node_positions[i+1][1] * height
                    
                    # Fade based on node values
                    intensity = (abs(self.nodes[i]) + abs(self.nodes[i+1])) / 2.0
                    pen.setColor(QColor(99, 102, 241, int(30 * intensity)))
                    painter.setPen(pen)
                    painter.drawLine(int(x1), int(y1), int(x2), int(y2))
                
                if i // size < size - 1:  # Bottom neighbor
                    x1 = self.node_positions[i][0] * width
                    y1 = self.node_positions[i][1] * height
                    x2 = self.node_positions[i+size][0] * width
                    y2 = self.node_positions[i+size][1] * height
                    
                    intensity = (abs(self.nodes[i]) + abs(self.nodes[i+size])) / 2.0
                    pen.setColor(QColor(99, 102, 241, int(30 * intensity)))
                    painter.setPen(pen)
                    painter.drawLine(int(x1), int(y1), int(x2), int(y2))
        
        # Draw nodes
        if len(self.nodes) > 0:
            max_node = max(abs(n) for n in self.nodes) if len(self.nodes) > 0 else 1.0
            if max_node == 0:
                max_node = 1.0
            
            for i, (pos, node_val) in enumerate(zip(self.node_positions, self.nodes)):
                x = pos[0] * width
                y = pos[1] * height
                z = pos[2]
                
                # Node size based on value
                node_size = 3 + abs(node_val) / max_node * 5
                
                # Color based on value and GCL
                if node_val > 0:
                    hue = 200  # Blue
                else:
                    hue = 320  # Pink
                
                saturation = int(200 + 55 * (1.0 - self.gcl))
                brightness = int(150 + 105 * abs(node_val) / max_node)
                alpha = int(150 + 105 * abs(node_val) / max_node)
                
                color = QColor.fromHsv(hue, saturation, brightness, alpha)
                
                # Add pulsing effect
                pulse = 1.0 + 0.1 * math.sin(self.animation_phase + i * 0.1)
                node_size *= pulse
                
                # Draw node with glow
                painter.setPen(Qt.NoPen)
                radial_grad = QRadialGradient(x, y, node_size * 2)
                radial_grad.setColorAt(0, color)
                radial_grad.setColorAt(1, QColor(color.red(), color.green(), color.blue(), 0))
                painter.setBrush(QBrush(radial_grad))
                painter.drawEllipse(QPointF(x, y), node_size * 2, node_size * 2)
                
                # Draw solid node
                painter.setBrush(QBrush(color))
                painter.drawEllipse(QPointF(x, y), node_size, node_size)
        
        # Draw GCL indicator
        gcl_text = f"GCL: {self.gcl:.3f}"
        painter.setPen(QPen(QColor(255, 255, 255)))
        font = QFont("Inter", 14, QFont.Bold)
        painter.setFont(font)
        painter.drawText(10, 30, gcl_text)


class MetricsChartWidget(QWidget):
    """Advanced real-time metrics chart"""
    
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.title = title
        self.setMinimumHeight(200)
        
        # Data series
        self.data_series = deque(maxlen=100)
        self.time_series = deque(maxlen=100)
        
        # Chart setup
        self.chart = QChart()
        self.chart.setTitle(title)
        self.chart.setTheme(QChart.ChartThemeDark)
        self.chart.legend().hide()
        self.chart.setBackgroundBrush(QBrush(QColor(15, 23, 42)))
        
        self.series = QLineSeries()
        self.chart.addSeries(self.series)
        
        # Axes
        self.axis_x = QValueAxis()
        self.axis_x.setRange(0, 100)
        self.axis_x.setLabelsColor(QColor(148, 163, 184))
        self.chart.addAxis(self.axis_x, Qt.AlignBottom)
        self.series.attachAxis(self.axis_x)
        
        self.axis_y = QValueAxis()
        self.axis_y.setRange(0, 1)
        self.axis_y.setLabelsColor(QColor(148, 163, 184))
        self.chart.addAxis(self.axis_y, Qt.AlignLeft)
        self.series.attachAxis(self.axis_y)
        
        # Chart view
        self.chart_view = QChartView(self.chart)
        self.chart_view.setRenderHint(QPainter.Antialiasing)
        self.chart_view.setBackgroundBrush(QBrush(QColor(15, 23, 42)))
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.chart_view)
    
    def add_data_point(self, value: float):
        """Add new data point"""
        self.data_series.append(value)
        self.time_series.append(len(self.data_series))
        
        # Update series
        self.series.clear()
        for i, (t, v) in enumerate(zip(self.time_series, self.data_series)):
            self.series.append(t, v)
        
        # Update axes
        if len(self.data_series) > 0:
            self.axis_x.setRange(0, max(100, len(self.data_series)))
            min_val = min(self.data_series) if self.data_series else 0
            max_val = max(self.data_series) if self.data_series else 1
            range_val = max_val - min_val
            if range_val == 0:
                range_val = 1
            self.axis_y.setRange(min_val - range_val * 0.1, max_val + range_val * 0.1)


class MetricCard(QWidget):
    """Professional metric display card"""
    
    def __init__(self, title: str, value: str = "0.00", unit: str = "", color: QColor = None, parent=None):
        super().__init__(parent)
        self.title = title
        self.value = value
        self.unit = unit
        self.color = color or QColor(99, 102, 241)
        
        self.setMinimumHeight(120)
        self.setStyleSheet(f"""
            QWidget {{
                background-color: rgba(15, 23, 42, 0.8);
                border: 1px solid rgba(99, 102, 241, 0.3);
                border-radius: 12px;
                padding: 15px;
            }}
        """)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(5)
        
        # Title
        title_label = QLabel(title)
        title_label.setStyleSheet("color: #94a3b8; font-size: 12px; font-weight: 500;")
        layout.addWidget(title_label)
        
        # Value
        self.value_label = QLabel(value)
        value_font = QFont("Inter", 28, QFont.Bold)
        self.value_label.setFont(value_font)
        self.value_label.setStyleSheet(f"color: rgb({self.color.red()}, {self.color.green()}, {self.color.blue()});")
        layout.addWidget(self.value_label)
        
        # Unit
        if unit:
            unit_label = QLabel(unit)
            unit_label.setStyleSheet("color: #64748b; font-size: 11px;")
            layout.addWidget(unit_label)
    
    def update_value(self, value: str):
        """Update displayed value"""
        self.value = value
        self.value_label.setText(value)


class AdvancedGoeckohGUI(QMainWindow):
    """Groundbreaking advanced GUI for Goeckoh System"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Goeckoh Neuro-Acoustic Exocortex - Advanced Interface")
        self.setGeometry(50, 50, 1920, 1080)
        
        # System state
        self.system_running = False
        self.orchestrator = None
        
        # Data buffers for visualization
        self.gcl_history = deque(maxlen=100)
        self.stress_history = deque(maxlen=100)
        self.latency_history = deque(maxlen=100)
        
        # Setup UI
        self.setup_ui()
        self.setup_styling()
        
        # Update timers
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_visualizations)
        self.update_timer.start(100)  # 10 Hz update rate
        
        # Simulate data for demo (remove in production)
        self.demo_timer = QTimer(self)
        self.demo_timer.timeout.connect(self.generate_demo_data)
        
    def setup_ui(self):
        """Setup the advanced UI layout"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Top bar with title and controls
        top_bar = self.create_top_bar()
        main_layout.addWidget(top_bar)
        
        # Main content area with splitter
        main_splitter = QSplitter(Qt.Horizontal)
        
        # Left panel: Visualizations
        left_panel = self.create_visualization_panel()
        main_splitter.addWidget(left_panel)
        
        # Right panel: Metrics and controls
        right_panel = self.create_metrics_panel()
        main_splitter.addWidget(right_panel)
        
        main_splitter.setStretchFactor(0, 2)
        main_splitter.setStretchFactor(1, 1)
        
        main_layout.addWidget(main_splitter)
        
        # Bottom status bar
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
        title_font = QFont("Inter", 24, QFont.Bold)
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
    
    def create_visualization_panel(self):
        """Create left panel with visualizations"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(15)
        
        # Voice waveform
        waveform_group = QGroupBox("Real-Time Voice Waveform")
        waveform_group.setStyleSheet(self.get_groupbox_style())
        waveform_layout = QVBoxLayout(waveform_group)
        
        self.waveform_widget = VoiceWaveformWidget()
        waveform_layout.addWidget(self.waveform_widget)
        
        layout.addWidget(waveform_group)
        
        # Crystalline Heart
        heart_group = QGroupBox("Crystalline Heart Lattice (1024 Nodes)")
        heart_group.setStyleSheet(self.get_groupbox_style())
        heart_layout = QVBoxLayout(heart_group)
        
        self.heart_widget = CrystallineHeartWidget()
        heart_layout.addWidget(self.heart_widget)
        
        layout.addWidget(heart_group)
        
        return widget
    
    def create_metrics_panel(self):
        """Create right panel with metrics"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(15)
        
        # Metric cards
        metrics_grid = QGridLayout()
        metrics_grid.setSpacing(15)
        
        self.gcl_card = MetricCard("Global Coherence Level", "1.000", "", QColor(99, 102, 241))
        metrics_grid.addWidget(self.gcl_card, 0, 0)
        
        self.stress_card = MetricCard("Stress Level", "0.000", "", QColor(236, 72, 153))
        metrics_grid.addWidget(self.stress_card, 0, 1)
        
        self.latency_card = MetricCard("Processing Latency", "0.0", "ms", QColor(139, 92, 246))
        metrics_grid.addWidget(self.latency_card, 1, 0)
        
        self.audio_level_card = MetricCard("Audio Level", "0.0", "dB", QColor(16, 185, 129))
        metrics_grid.addWidget(self.audio_level_card, 1, 1)
        
        layout.addLayout(metrics_grid)
        
        # Charts
        self.gcl_chart = MetricsChartWidget("GCL History")
        layout.addWidget(self.gcl_chart)
        
        self.stress_chart = MetricsChartWidget("Stress History")
        layout.addWidget(self.stress_chart)
        
        # Input area
        input_group = QGroupBox("System Input")
        input_group.setStyleSheet(self.get_groupbox_style())
        input_layout = QVBoxLayout(input_group)
        
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Enter text or command...")
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
        
        send_btn = QPushButton("Send")
        send_btn.setStyleSheet(self.get_button_style("primary"))
        send_btn.clicked.connect(self.process_input)
        input_layout.addWidget(send_btn)
        
        layout.addWidget(input_group)
        
        return widget
    
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
        
        self.status_label = QLabel("● System Status: Ready")
        self.status_label.setStyleSheet("color: #10b981; font-weight: 600;")
        layout.addWidget(self.status_label)
        
        layout.addStretch()
        
        self.connection_label = QLabel("Audio: Not Connected")
        self.connection_label.setStyleSheet("color: #f59e0b;")
        layout.addWidget(self.connection_label)
        
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
        self.status_label.setText("● System Status: Running")
        self.status_label.setStyleSheet("color: #10b981; font-weight: 600;")
        
        # Try to start actual system
        try:
            from system_launcher import SystemOrchestrator, SystemConfig
            config = SystemConfig(mode="universe")
            self.orchestrator = SystemOrchestrator(config)
            # Start in background thread
            import threading
            thread = threading.Thread(target=self.orchestrator.run_interactive, daemon=True)
            thread.start()
        except Exception as e:
            print(f"Could not start full system: {e}")
        
        # Start demo data generation
        self.demo_timer.start(100)
    
    def stop_system(self):
        """Stop the system"""
        if not self.system_running:
            return
        
        self.system_running = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("● System Status: Stopped")
        self.status_label.setStyleSheet("color: #ef4444; font-weight: 600;")
        
        self.demo_timer.stop()
        
        if self.orchestrator:
            try:
                self.orchestrator.stop_system()
            except:
                pass
    
    def process_input(self):
        """Process user input"""
        text = self.input_field.text().strip()
        if not text:
            return
        
        self.input_field.clear()
        # Process input here
        print(f"Processing: {text}")
    
    def generate_demo_data(self):
        """Generate demo data for visualization (remove in production)"""
        if not self.system_running:
            return
        
        # Generate random audio data
        audio_sample = np.sin(time.time() * 2 * np.pi * 440) * 0.5 + np.random.normal(0, 0.1)
        self.waveform_widget.add_audio_sample(audio_sample)
        
        # Generate spectrum
        spectrum = np.abs(np.fft.fft(np.random.randn(512)))
        self.waveform_widget.add_spectrum(spectrum)
        
        # Update heart lattice
        nodes = np.random.normal(0, 0.1, 1024).astype(np.float32)
        gcl = max(0, min(1, 0.7 + np.random.normal(0, 0.05)))
        stress = 1.0 - gcl
        self.heart_widget.update_lattice(nodes, gcl, stress)
        
        # Update metrics
        self.gcl_history.append(gcl)
        self.stress_history.append(stress)
        self.latency_history.append(np.random.uniform(10, 50))
        
        self.gcl_card.update_value(f"{gcl:.3f}")
        self.stress_card.update_value(f"{stress:.3f}")
        self.latency_card.update_value(f"{self.latency_history[-1]:.1f}")
        self.audio_level_card.update_value(f"{20 * np.log10(abs(audio_sample) + 0.001):.1f}")
        
        self.gcl_chart.add_data_point(gcl)
        self.stress_chart.add_data_point(stress)
    
    def update_visualizations(self):
        """Update all visualizations"""
        # This is called by the timer
        # Real updates would come from system data
        pass


def main():
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Goeckoh Advanced GUI")
    app.setOrganizationName("Goeckoh")
    
    # Set high DPI scaling
    app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    window = AdvancedGoeckohGUI()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()


