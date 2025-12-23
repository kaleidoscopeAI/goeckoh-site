import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

class EnhancedVisualizer:
    def __init__(self):
        self.viz_json = None

    def update(self, ai_system):
        n = len(ai_system.cube.nodes)
        pos = np.array([node.position for node in ai_system.cube.nodes])
        arousal = np.array([node.arousal for node in ai_system.cube.nodes])
        valence = np.array([node.valence for node in ai_system.cube.nodes])

        fig = make_subplots(
            rows=3, cols=4,
            specs=[
                [{'type':'scatter3d'}, {'type':'scatter3d'}, {'type':'polar'}, {'type':'heatmap'}],
                [{'type':'scatter'}, {'type':'bar'}, {'type':'scatter'}, None],
                [{'type':'scatter'}, None, None, None]
            ],
            subplot_titles=["3D Network", "Position Lattice", "Emotional Radar", "Energy Heatmap",
                            "Health Over Time", "Energy Flow", "Device Control Signal", ""])

        # Node Scatter3D, Edge Lines
        fig.add_trace(go.Scatter3d(
            x=pos[:,0], y=pos[:,1], z=pos[:,2],
            mode='markers',
            marker=dict(color=arousal, colorscale='Viridis', size=6),
            name='Nodes'), row=1, col=1)

        for edge in ai_system.cube.graph.edges():
            start, end = pos[edge[0]], pos[edge[1]]
            fig.add_trace(go.Scatter3d(
                x=[start[0], end[0]], y=[start[1], end[1]], z=[start[2], end[2]],
                mode='lines', line=dict(color='gray', width=2), showlegend=False), row=1, col=1)

        # Emotional Radar using avg valence/arousal/dominance
        fig.add_trace(go.Scatterpolar(
            r=[np.mean(valence), np.mean(arousal), np.mean([n.dominance for n in ai_system.cube.nodes])],
            theta=['Valence', 'Arousal', 'Dominance'],
            fill='toself',
            name='Emotion'), row=1, col=3)

        # Heatmap of node energies
        energies = np.array([n.energy for n in ai_system.cube.nodes])
        heatmap_mat = energies.reshape(int(np.sqrt(n)), -1)
        fig.add_trace(go.Heatmap(z=heatmap_mat, colorscale='Electric'), row=1, col=4)

        # Health/Coherence Timeline
        fig.add_trace(go.Scatter(
            y=ai_system.metrics.health_log,
            mode='lines+markers',
            name='Health Over Time'), row=2, col=1)

        # Energy Flow bar chart (top nodes)
        fig.add_trace(go.Bar(
            y=energies[:20],
            name='Top Node Energy'), row=2, col=2)

        # Device control signal (time series)
        if hasattr(ai_system.device_controller, "signal_log"):
            sig_log = ai_system.device_controller.signal_log
            fig.add_trace(go.Scatter(
                y=sig_log,
                mode='lines',
                name='Device Control Signal'), row=2, col=3)

        fig.update_layout(height=950, showlegend=True)
        self.viz_json = fig.to_json()
