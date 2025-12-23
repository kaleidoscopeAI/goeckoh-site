def __init__(self, ai_system):
    self.ai_system = ai_system

def generate_dashboard_json(self):
    n = len(self.ai_system.nodes)
    pos = np.array([node.position.components for node in self.ai_system.nodes])
    arousal = np.array([node.emotional_state.arousal for node in self.ai_system.nodes])

    fig = make_subplots(rows=2, cols=2, specs=[[{'type':'scatter3d'}, {'type':'scatter'}], [{'type':'heatmap'}, None]])

    fig.add_trace(go.Scatter3d(x=pos[:,0], y=pos[:,1], z=pos[:,2], mode='markers',
                               marker=dict(color=arousal, colorscale='Viridis', size=5),
                               name='Nodes'), row=1, col=1)

    fig.add_trace(go.Scatter(x=np.arange(n), y=arousal, mode='lines', name='Arousal'), row=1, col=2)

    fig.add_trace(go.Heatmap(z=arousal.reshape(int(np.sqrt(n)), -1), colorscale='Viridis'), row=2, col=1)

    return fig.to_json()
