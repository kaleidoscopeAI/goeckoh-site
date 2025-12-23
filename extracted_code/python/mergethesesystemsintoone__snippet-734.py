@staticmethod
def create_dashboard(insights, nodes):
    app = Dash(__name__)

    app.layout = html.Div([
        html.H1("Unified Node System Dashboard"),
        dcc.Graph(
            id="data-vs-insights",
            figure={
                "data": [
                    go.Pie(labels=["Raw Data", "Insights"], values=[len(nodes), len(insights)], hole=0.4)
                ],
                "layout": {"title": "Data vs Insights"}
            }
        ),
        dcc.Graph(
            id="node-energy",
            figure={
                "data": [
                    go.Bar(
                        x=[node.node_id for node in nodes],
                        y=[node.energy for node in nodes]
                    )
                ],
                "layout": {"title": "Node Energy Levels"}
            }
        )
    ])

    # Disable debug mode in production
    app.run_server(debug=False)

