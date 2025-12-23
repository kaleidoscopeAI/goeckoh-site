class ConsciousCube:
    """
    Represents the Conscious Cube, the living, self-architecting embodiment of Kaleidoscope AI's intelligence.
    Its internal graph structure dynamically evolves based on incoming data and Super Node interactions.
    """
    def __init__(self):
        self.graph = nx.Graph()
        self.node_states = {}  # Stores current "activity" or "insight_strength" for conceptual nodes
        self.viz_map = {}  # Conceptual mapping for visualization properties

        logging.info("--- Conscious Cube Initialized: Awaiting self-architecture directives. ---")

    def add_conceptual_node(self, node_id, initial_strength=0.1):
        """
        Adds a new conceptual node (e.g., a synthesized insight, a Super Node connection point).
        Robustness: Validate node_id type and initial_strength range.
        """
        if not isinstance(node_id, str) or not node_id:
            logging.error("Cube Directive: node_id must be a non-empty string.")
            return
        if not isinstance(initial_strength, (int, float)) or not (0.0 <= initial_strength <= 1.0):
            logging.error("Cube Directive: initial_strength must be a float between 0.0 and 1.0.")
            return

        if node_id not in self.graph.nodes():
            self.graph.add_node(node_id)
            self.node_states[node_id] = {'strength': initial_strength, 'activity_score': 0.0}
            self._update_viz_properties(node_id)
            logging.info(f"Cube Directive: Added conceptual node '{node_id}'.")
        else:
            logging.warning(f"Node '{node_id}' already exists. Skipping addition.")

    def update_node_activity(self, node_id, activity_score):
        """
        Updates a node's 'activity_score' based on incoming data/Super Node interaction.
        This drives structural changes and conceptual visualization.
        Robustness: Validate node_id and activity_score.
        """
        if not isinstance(node_id, str) or not node_id:
            logging.error("Cube Directive: node_id must be a non-empty string.")
            return
        if not isinstance(activity_score, (int, float)) or not (0.0 <= activity_score <= 1.0):
            logging.error("Cube Directive: activity_score must be a float between 0.0 and 1.0.")
            return

        if node_id in self.node_states:
            self.node_states[node_id]['activity_score'] = activity_score
            # Smooth update for strength based on activity
            self.node_states[node_id]['strength'] = np.clip(
                self.node_states[node_id]['strength'] * 0.9 + activity_score * 0.1, 0.0, 1.0
            )
            self._update_viz_properties(node_id)
            logging.info(f"Cube Directive: Updated '{node_id}' activity to {activity_score:.2f} (Strength: {self.node_states[node_id]['strength']:.2f}).")
            # Trigger self-architecture based on this update
            self._self_architect_based_on_activity(node_id)
        else:
            logging.warning(f"Node '{node_id}' not found. Cannot update activity. Consider adding it first.")

    def _self_architect_based_on_activity(self, central_node_id):
        """
        Simplified "self-architecting" rule: High activity in a node might strengthen connections
        or form new "insight hubs" (conceptual links). Inactive nodes might be pruned.
        This represents the 'recursive optimization engine' for its topology.
        Robustness: Handle potential graph modification errors.
        """
        if central_node_id not in self.node_states:
            logging.warning(f"Self-Architecture: Central node '{central_node_id}' not found in states. Skipping.")
            return

        try:
            if self.node_states[central_node_id]['activity_score'] > 0.7:  # High activity threshold
                logging.info(f"\nCube Directive (Self-Architecture): High activity detected in '{central_node_id}'. Initiating structural refinement.")
                # Use list() to iterate over a copy of nodes, allowing modification during iteration
                for other_node_id in list(self.graph.nodes()):
                    if other_node_id != central_node_id:
                        # Strengthen existing connections based on activity
                        if self.graph.has_edge(central_node_id, other_node_id):
                            current_weight = self.graph[central_node_id][other_node_id].get('weight', 0.5)
                            new_weight = np.clip(current_weight + 0.1, 0.0, 1.0)
                            self.graph[central_node_id][other_node_id]['weight'] = new_weight
                            logging.info(f"    Strengthened edge {central_node_id}-{other_node_id} to {new_weight:.2f}.")
                            self._update_edge_viz_properties(central_node_id, other_node_id)

                        # Novel Choice: "Emergent Grammar of Connectivity" - create new links based on inferred semantic proximity
                        # (simplified: if both nodes are strong and not connected, and random chance)
                        if (other_node_id in self.node_states and  # Ensure other_node_id still exists
                            self.node_states[central_node_id]['strength'] > 0.6 and
                            self.node_states[other_node_id]['strength'] > 0.6 and
                            not self.graph.has_edge(central_node_id, other_node_id) and
                            random.random() < 0.2):  # Probabilistic new link formation
                            logging.info(f"    Forming new emergent link between '{central_node_id}' and '{other_node_id}'.")
                            self.graph.add_edge(central_node_id, other_node_id, weight=0.5)  # Initial weight
                            self._update_edge_viz_properties(central_node_id, other_node_id)

            # "New Math" - Dynamic Topology Pruning: If a node is consistently inactive and isolated, it might be pruned.
            # Robustness: Ensure node exists before attempting removal.
            for node_id, state in list(self.node_states.items()):
                if state['strength'] < 0.1 and len(self.graph.edges(node_id)) == 0 and node_id != central_node_id:
                    if node_id in self.graph:
                        self.graph.remove_node(node_id)
                        del self.node_states[node_id]
                        if node_id in self.viz_map: del self.viz_map[node_id]
                        logging.info(f"  Pruned inactive and isolated node '{node_id}'.")
        except Exception as e:
            logging.error(f"Self-Architecture encountered an error: {e}")

    def _update_viz_properties(self, node_id):
        """Conceptual mapping of node state to visualization properties (color, size, form)."""
        if node_id not in self.node_states:  # Robustness: Ensure node state exists
            return
        strength = self.node_states[node_id]['strength']
        activity = self.node_states[node_id]['activity_score']
        self.viz_map[node_id] = {
            'color': self._get_color_from_activity(activity),
            'size': 0.5 + strength * 1.5,  # Size from 0.5 to 2.0
            'form': 'sphere' if strength < 0.7 else 'pyramid_active'  # Dynamic form
        }

    def _update_edge_viz_properties(self, u, v):
        """Conceptual mapping of edge weight to visualization properties (thickness, glow)."""
        if self.graph.has_edge(u, v):  # Robustness: Ensure edge exists
            weight = self.graph[u][v].get('weight', 0.0)
            self.viz_map[(u, v)] = {
                'thickness': 0.1 + weight * 0.9,  # Thickness from 0.1 to 1.0
                'glow': True if weight > 0.8 else False
            }
        else:
            logging.warning(f"Attempted to update viz for non-existent edge {u}-{v}.")

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
                'thickness': edge_data.get('weight', 0.5),  # Default if not explicitly updated
                'glow': False
            })
        return {"nodes": nodes_viz, "edges": edges_viz}

    def report_status(self):
        """Prints the current status of the Conscious Cube."""
        logging.info("\n--- Conscious Cube Current State ---")
        logging.info("  Nodes:")
        if not self.node_states:
            logging.info("    No nodes currently in the cube.")
        for node_id, state in self.node_states.items():
            logging.info(f"    - {node_id}: Strength={state['strength']:.2f}, Activity={state['activity_score']:.2f}")
        logging.info("  Edges (top 5 by weight):")
        if not self.graph.edges():
            logging.info("    No edges currently in the cube.")
        sorted_edges = sorted(self.graph.edges(data=True), key=lambda x: x[2].get('weight', 0), reverse=True)
        for u, v, data in sorted_edges[:5]:
            logging.info(f"    - {u} <-> {v}: Weight={data.get('weight', 0):.2f}")
        logging.info("------------------------------------")


