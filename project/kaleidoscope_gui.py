import sys
import threading
import time
import random
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QListWidget
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# This is a placeholder for the actual AI components.
# In a real application, these would be imported from your other modules.
from backend.core.node import Node
from backend.data.membrane import allocate_nodes
from backend.engines.kaleidoscope_engine import KaleidoscopeEngine
from backend.engines.perspective_engine import PerspectiveEngine
from backend.memory.graph import MemoryGraph
from backend.core.phi_calculator import PhiCalculator
from backend.core.hamiltonian import HamiltonianController
from backend.utils.safety.predictive_safety import PredictiveSafetySystem
from backend.utils.ethics.consent_manager import ConsentManager

# Simulation Wrapper for Threading
class SimulationThread(threading.Thread):
    def __init__(self, nodes, engines, memory_graph, phi_calc, hamiltonian, safety_system, consent_mgr, update_callback):
        self.k_engine, self.p_engine = engines
        super().__init__()
        self.nodes = nodes
        self.memory_graph = memory_graph
        self.phi_calc = phi_calc
        self.safety_system = safety_system
        self.update_callback = update_callback
        self.hamiltonian = hamiltonian
        self.consent_mgr = consent_mgr
        self.running = True

    def run(self):
        while self.running:
            insights = []
            speculative = []
            edges = []

            # Node actions
            for node in self.nodes:
                if not self.consent_mgr.check_consent(node.node_id):
                    continue
                node_insight = node.act(self.nodes)
                insights.append(node_insight)

                if self.safety_system.intervene(node):
                    print(f"[SAFETY] Intervention applied to {node.node_id}")

                self.memory_graph.add_node(node)

            # Engine outputs
            insights += self.k_engine.process(self.nodes)
            speculative += self.p_engine.process(self.nodes)

            # Hamiltonian updates
            for node in self.nodes:
                H = self.hamiltonian.compute(node, self.nodes)
                node.energy += 0.01 - 0.001*H
                node.energy = max(0.0, min(1.0, node.energy))

            # Node replication
            new_nodes = []
            for node in self.nodes:
                clone = node.replicate()
                if clone:
                    self.consent_mgr.give_consent(clone.node_id)
                    self.memory_graph.add_node(clone)
                    new_nodes.append(clone)
            self.nodes += new_nodes

            # Build edges for bond network
            node_list = list(self.memory_graph.graph.nodes())
            for i in range(len(node_list)):
                for j in range(i + 1, len(node_list)):
                    if random.random() < 0.02: # random connections for visualization
                        edges.append((i, j))

            # Compute metrics
            avg_energy = sum(n.energy for n in self.nodes) / len(self.nodes)
            avg_phi = sum(self.phi_calc.compute_phi(n, self.nodes) for n in self.nodes) / len(self.nodes)

            # Callback to update GUI
            self.update_callback(self.nodes, insights, speculative, edges, avg_energy, avg_phi)
            time.sleep(0.1)

# GUI
class KaleidoscopeGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Kaleidoscope AI Dashboard")
        self.setGeometry(50, 50, 1400, 800)
        layout = QHBoxLayout()
        self.setLayout(layout)

        # Left panel: 3D node visualization
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas, 3)

        # Right panel: metrics + controls
        right_panel = QVBoxLayout()
        layout.addLayout(right_panel, 1)

        # Metrics labels
        self.energy_label = QLabel("Avg Energy: 0.0")
        self.phi_label = QLabel("Avg Phi: 0.0")
        right_panel.addWidget(self.energy_label)
        right_panel.addWidget(self.phi_label)

        # Insights list
        self.insights_list = QListWidget()
        self.speculative_list = QListWidget()
        right_panel.addWidget(QLabel("Insights:"))
        right_panel.addWidget(self.insights_list)
        right_panel.addWidget(QLabel("Speculative Insights:"))
        right_panel.addWidget(self.speculative_list)

        # Control buttons
        self.start_button = QPushButton("Start Simulation")
        self.stop_button = QPushButton("Stop Simulation")
        right_panel.addWidget(self.start_button)
        right_panel.addWidget(self.stop_button)

        # Simulation objects
        num_nodes = allocate_nodes(500, 10)
        self.nodes = [Node(f"N{i}") for i in range(num_nodes)]
        self.consent_mgr = ConsentManager()
        for node in self.nodes:
            self.consent_mgr.give_consent(node.node_id)

        self.k_engine = KaleidoscopeEngine()
        self.p_engine = PerspectiveEngine()
        self.memory_graph = MemoryGraph()
        for node in self.nodes:
            self.memory_graph.add_node(node)

        self.phi_calc = PhiCalculator()
        self.hamiltonian = HamiltonianController()
        self.safety_system = PredictiveSafetySystem()

        self.sim_thread = SimulationThread(
            self.nodes,
            (self.k_engine, self.p_engine),
            self.memory_graph,
            self.phi_calc,
            self.hamiltonian,
            self.safety_system,
            self.consent_mgr,
            self.update_gui
        )

        # Connect buttons
        self.start_button.clicked.connect(self.start_simulation)
        self.stop_button.clicked.connect(self.stop_simulation)

        # Timer to refresh canvas
        self.timer = QTimer()
        self.timer.timeout.connect(self.refresh_plot)
        self.timer.start(200)

    def start_simulation(self):
        if not self.sim_thread.is_alive():
            self.sim_thread.running = True
            self.sim_thread.start()

    def stop_simulation(self):
        self.sim_thread.running = False

    def update_gui(self, nodes, insights, speculative, edges, avg_energy, avg_phi):
        self.current_nodes = nodes
        self.current_edges = edges
        self.current_insights = insights
        self.current_speculative = speculative
        self.energy_label.setText(f"Avg Energy: {avg_energy:.3f}")
        self.phi_label.setText(f"Avg Phi: {avg_phi:.3f}")
        self.insights_list.clear()
        self.insights_list.addItems([str(i) for i in insights[:10]])
        self.speculative_list.clear()
        self.speculative_list.addItems([str(i) for i in speculative[:10]])

    def refresh_plot(self):
        if not hasattr(self, 'current_nodes'):
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111, projection='3d')
        ax.set_facecolor('black')
        ax.grid(False)

        xs, ys, zs, cs = [], [], [], []
        for node in self.current_nodes:
            xs.append(random.uniform(-1, 1))
            ys.append(random.uniform(-1, 1))
            zs.append(random.uniform(-1, 1))
            c = (1 - node.energy, node.energy, 0.0)
            cs.append(c)

        ax.scatter(xs, ys, zs, c=cs, s=50)
        self.canvas.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = KaleidoscopeGUI()
    gui.show()
    sys.exit(app.exec_())
