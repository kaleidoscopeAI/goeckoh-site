#!/usr/bin/env python3
"""
Simple PySide6 GUI for Goeckoh System
Fallback GUI when goeckoh_gui module is not available
"""

import sys
from pathlib import Path
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                              QHBoxLayout, QLabel, QPushButton, QTextEdit, QLineEdit)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QPalette, QColor

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

class GoeckohMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Goeckoh Neuro-Acoustic Exocortex")
        self.setGeometry(100, 100, 1200, 800)
        
        # Set dark theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0f172a;
            }
            QWidget {
                background-color: #0f172a;
                color: #e2e8f0;
                font-family: 'Inter', sans-serif;
            }
            QLabel {
                color: #e2e8f0;
            }
            QPushButton {
                background-color: #6366f1;
                color: white;
                border: none;
                padding: 10px 20px;
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
            QTextEdit, QLineEdit {
                background-color: #1e293b;
                border: 1px solid #334155;
                border-radius: 8px;
                padding: 10px;
                color: #e2e8f0;
                font-size: 14px;
            }
            QLineEdit:focus {
                border: 2px solid #6366f1;
            }
        """)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        layout = QVBoxLayout(central_widget)
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)
        
        # Title
        title = QLabel("Goeckoh Neuro-Acoustic Exocortex")
        title.setAlignment(Qt.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(24)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setStyleSheet("color: #6366f1; margin-bottom: 10px;")
        layout.addWidget(title)
        
        # Subtitle
        subtitle = QLabel("Therapeutic Voice Therapy & AGI System")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: #94a3b8; font-size: 16px; margin-bottom: 20px;")
        layout.addWidget(subtitle)
        
        # Status label
        self.status_label = QLabel("System Status: Initializing...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #f59e0b; font-size: 14px; padding: 10px;")
        layout.addWidget(self.status_label)
        
        # Output area
        output_label = QLabel("System Output:")
        output_label.setStyleSheet("color: #e2e8f0; font-weight: 600; font-size: 14px;")
        layout.addWidget(output_label)
        
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setMinimumHeight(400)
        layout.addWidget(self.output_text)
        
        # Input area
        input_layout = QHBoxLayout()
        
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Enter command or text...")
        self.input_field.returnPressed.connect(self.process_input)
        input_layout.addWidget(self.input_field)
        
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.process_input)
        input_layout.addWidget(self.send_button)
        
        layout.addLayout(input_layout)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.start_button = QPushButton("Start System")
        self.start_button.clicked.connect(self.start_system)
        button_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("Stop System")
        self.stop_button.clicked.connect(self.stop_system)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.stop_button)
        
        layout.addLayout(button_layout)
        
        # System state
        self.system_running = False
        self.update_status("Ready - Click 'Start System' to begin")
        
    def update_status(self, message):
        """Update status label"""
        self.status_label.setText(f"System Status: {message}")
        
    def add_output(self, text):
        """Add text to output area"""
        self.output_text.append(text)
        # Auto-scroll to bottom
        scrollbar = self.output_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
    def process_input(self):
        """Process user input"""
        text = self.input_field.text().strip()
        if not text:
            return
            
        self.add_output(f"> {text}")
        self.input_field.clear()
        
        # Try to process with system
        try:
            # Import and use system components if available
            self.add_output("Processing input...")
            # Placeholder - integrate with actual system
            self.add_output("System response: Input received and processed.")
        except Exception as e:
            self.add_output(f"Error: {str(e)}")
    
    def start_system(self):
        """Start the system"""
        if self.system_running:
            return
            
        self.system_running = True
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.update_status("Running")
        self.add_output("=" * 60)
        self.add_output("Goeckoh System Starting...")
        self.add_output("=" * 60)
        
        # Try to import and start actual system
        try:
            self.add_output("Loading system components...")
            # Try to import system launcher
            try:
                from system_launcher import SystemOrchestrator, SystemConfig
                config = SystemConfig(mode="universe")
                self.orchestrator = SystemOrchestrator(config)
                self.add_output("System components loaded successfully!")
                self.add_output("System is ready for voice therapy sessions.")
            except ImportError as e:
                self.add_output(f"Note: Some system components not available: {e}")
                self.add_output("Running in basic mode.")
        except Exception as e:
            self.add_output(f"Warning: {str(e)}")
            self.add_output("System running in limited mode.")
    
    def stop_system(self):
        """Stop the system"""
        if not self.system_running:
            return
            
        self.system_running = False
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.update_status("Stopped")
        self.add_output("=" * 60)
        self.add_output("System Stopped")
        self.add_output("=" * 60)
        
        # Stop orchestrator if it exists
        if hasattr(self, 'orchestrator'):
            try:
                self.orchestrator.stop_system()
            except:
                pass

def main():
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Goeckoh System")
    app.setOrganizationName("Goeckoh")
    
    window = GoeckohMainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()


