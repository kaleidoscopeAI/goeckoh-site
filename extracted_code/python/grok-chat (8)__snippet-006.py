class ConsciousCube:
    """
    Represents the Conscious Cube, the living, self-architecting embodiment of Kaleidoscope AI's intelligence.
    Its internal graph structure dynamically evolves based on incoming data and Super Node interactions.
    """
    def __init__(self):
        self.graph = nx.Graph()
        self.node_states = {} # Stores current "activity" or "insight_strength" for conceptual nodes
        self.viz_map = {} # Conceptual mapping for visualization properties

        print("--- Conscious Cube Initialized: Awaiting self-architecture directives. ---")

    def add_conceptual_node(self, node_id, initial_strength=0.1):
        """
        Adds a new conceptual node (e.g., a synthesized insight, a Super Node connection point).
        Robustness: Validate node_id type and initial_strength range.
        """
        if not isinstance(node_id, str) or not node_id:
            print("  [ERROR] Cube Directive: node_id must be a non-empty string.")
            return
        if not isinstance(initial_strength, (int, float)) or not (0.0 <= initial_strength <= 1.0):
            print("  [ERROR] Cube Directive: initial_strength must be a float between 0.0 and 1.0.")
            return

        if node_id not in self.graph.nodes():
            self.graph.add_node(node_id)
            self.node_states[node_id] = {'strength': initial_strength, 'activity_score': 0.0}
            self._update_viz_properties(node_id)
            print(f"  Cube Directive: Added conceptual node '{node_id}'.")
        else:
            print(f"  Node '{node_id}' already exists. Skipping addition.")

    def update_node_activity(self, node_id, activity_score):
        """
        Updates a node's 'activity_score' based on incoming data/Super Node interaction.
        This drives structural changes and conceptual visualization.
        Robustness: Validate node_id and activity_score.
        """
        if not isinstance(node_id, str) or not node_id:
            print("  [ERROR] Cube Directive: node_id must be a non-empty string.")
            return
        if not isinstance(activity_score, (int, float)) or not (0.0 <= activity_score <= 1.0):
            print("  [ERROR] Cube Directive: activity_score must be a float between 0.0 and 1.0.")
            return

        if node_id in self.node_states:
            self.node_states[node_id]['activity_score'] = activity_score
            # Smooth update for strength based on activity
            self.node_states[node_id]['strength'] = np.clip(
                self.node_states[node_id]['strength'] * 0.9 + activity_score * 0.1, 0.0, 1.0
            )
            self._update_viz_properties(node_id)
            print(f"  Cube Directive: Updated '{node_id}' activity to {activity_score:.2f} (Strength: {self.node_states[node_id]['strength']:.2f}).")
            # Trigger self-architecture based on this update
            self._self_architect_based_on_activity(node_id)
        else:
            print(f"  [WARNING] Node '{node_id}' not found. Cannot update activity. Consider adding it first.")

    def _self_architect_based_on_activity(self, central_node_id):
        """
        Simplified "self-architecting" rule: High activity in a node might strengthen connections
        or form new "insight hubs" (conceptual links). Inactive nodes might be pruned.
        This represents the 'recursive optimization engine' for its topology.
        Robustness: Handle potential graph modification errors.
        """
        if central_node_id not in self.node_states:
            print(f"  [WARNING] Self-Architecture: Central node '{central_node_id}' not found in states. Skipping.")
            return

        try:
            if self.node_states[central_node_id]['activity_score'] > 0.7: # High activity threshold
                print(f"\n  Cube Directive (Self-Architecture): High activity detected in '{central_node_id}'. Initiating structural refinement.")
                # Use list() to iterate over a copy of nodes, allowing modification during iteration
                for other_node_id in list(self.graph.nodes()):
                    if other_node_id != central_node_id:
                        # Strengthen existing connections based on activity
                        if self.graph.has_edge(central_node_id, other_node_id):
                            current_weight = self.graph[central_node_id][other_node_id].get('weight', 0.5)
                            new_weight = np.clip(current_weight + 0.1, 0.0, 1.0)
                            self.graph[central_node_id][other_node_id]['weight'] = new_weight
                            print(f"    Strengthened edge {central_node_id}-{other_node_id} to {new_weight:.2f}.")
                            self._update_edge_viz_properties(central_node_id, other_node_id)

                        # Novel Choice: "Emergent Grammar of Connectivity" - create new links based on inferred semantic proximity
                        # (simplified: if both nodes are strong and not connected, and random chance)
                        if (other_node_id in self.node_states and # Ensure other_node_id still exists
                            self.node_states[central_node_id]['strength'] > 0.6 and
                            self.node_states[other_node_id]['strength'] > 0.6 and
                            not self.graph.has_edge(central_node_id, other_node_id) and
                            random.random() < 0.2): # Probabilistic new link formation
                            print(f"    Forming new emergent link between '{central_node_id}' and '{other_node_id}'.")
                            self.graph.add_edge(central_node_id, other_node_id, weight=0.5) # Initial weight
                            self._update_edge_viz_properties(central_node_id, other_node_id)

            # "New Math" - Dynamic Topology Pruning: If a node is consistently inactive and isolated, it might be pruned.
            # Robustness: Ensure node exists before attempting removal.
            for node_id, state in list(self.node_states.items()):
                if state['strength'] < 0.1 and len(self.graph.edges(node_id)) == 0 and node_id != central_node_id:
                    if node_id in self.graph:
                        self.graph.remove_node(node_id)
                        del self.node_states[node_id]
                        if node_id in self.viz_map: del self.viz_map[node_id]
                        print(f"  Pruned inactive and isolated node '{node_id}'.")
        except Exception as e:
            print(f"  [ERROR] Self-Architecture encountered an error: {e}")

    def _update_viz_properties(self, node_id):
        """Conceptual mapping of node state to visualization properties (color, size, form)."""
        if node_id not in self.node_states: # Robustness: Ensure node state exists
            return
        strength = self.node_states[node_id]['strength']
        activity = self.node_states[node_id]['activity_score']
        self.viz_map[node_id] = {
            'color': self._get_color_from_activity(activity),
            'size': 0.5 + strength * 1.5, # Size from 0.5 to 2.0
            'form': 'sphere' if strength < 0.7 else 'pyramid_active' # Dynamic form
        }

    def _update_edge_viz_properties(self, u, v):
        """Conceptual mapping of edge weight to visualization properties (thickness, glow)."""
        if self.graph.has_edge(u, v): # Robustness: Ensure edge exists
            weight = self.graph[u][v].get('weight', 0.0)
            self.viz_map[(u, v)] = {
                'thickness': 0.1 + weight * 0.9, # Thickness from 0.1 to 1.0
                'glow': True if weight > 0.8 else False
            }
        else:
            print(f"  [WARNING] Attempted to update viz for non-existent edge {u}-{v}.")

    def _get_color_from_activity(self, activity):
        """Simple conceptual color mapping."""
        if activity < 0.4: return "green"
        elif activity < 0.7: return "yellow"
        else: return "red"

    def get_current_viz_state(self):
        """Returns the current conceptual visualization state for rendering."""
        nodes_viz = {node_id: self.viz_map.get(node_id, {'color': 'grey', 'size': 1.0, 'form': 'sphere'}) for node_id in self.graph.nodes()}
        edges_viz = {}
        for u, v in self.graph.edges():
            edge_data = self.graph[u][v]
            edges_viz[(u, v)] = self.viz_map.get((u, v), {
                'thickness': edge_data.get('weight', 0.5), # Default if not explicitly updated
                'glow': False
            })
        return {"nodes": nodes_viz, "edges": edges_viz}

    def report_status(self):
        """Prints the current status of the Conscious Cube."""
        print("\n--- Conscious Cube Current State ---")
        print("  Nodes:")
        if not self.node_states:
            print("    No nodes currently in the cube.")
        for node_id, state in self.node_states.items():
            print(f"    - {node_id}: Strength={state['strength']:.2f}, Activity={state['activity_score']:.2f}")
        print("  Edges (top 5 by weight):")
        if not self.graph.edges():
            print("    No edges currently in the cube.")
        sorted_edges = sorted(self.graph.edges(data=True), key=lambda x: x[2].get('weight', 0), reverse=True)
        for u, v, data in sorted_edges[:5]:
            print(f"    - {u} <-> {v}: Weight={data.get('weight', 0):.2f}")
        print("------------------------------------")

    def visualize_voxels(self):
        """Visualizes the Conscious Cube using voxel-based representation."""
        if not self.graph.nodes:
            print("No nodes to visualize.")
            return

        # Assign random 3D positions to nodes for visualization
        positions = {node: np.random.randint(0, 10, size=3) for node in self.graph.nodes()}
        grid_size = 10
        voxels = np.zeros((grid_size, grid_size, grid_size), dtype=bool)

        for pos in positions.values():
            x, y, z = pos
            voxels[x, y, z] = True

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.voxels(voxels, facecolors='blue', edgecolor='k')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title('Voxel Visualization of Conscious Cube')
        plt.show()


