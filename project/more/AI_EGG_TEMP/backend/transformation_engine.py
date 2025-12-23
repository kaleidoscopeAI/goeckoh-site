# backend/transformation_engine.py
import logging

class TransformationEngine:
    def __init__(self, simulator, memory_store):
        self.sim = simulator
        self.memory = memory_store
        logging.info("TransformationEngine initialized.")

    def handle_events(self, events):
        for ev in events:
            etype = ev.get('type')
            if etype == 'bond_broken':
                n1, n2 = ev['nodes']
                self.sim.remove_bond(n1, n2)
                self.memory.add_event(f"Bond {n1}-{n2} broke")
                logging.info(f"Removed broken bond between Node {n1} and Node {n2}")

    def apply_suggestions(self, suggestions):
        for action in suggestions:
            act_type = action.get('action')
            try:
                if act_type == 'add_bond':
                    nodes = action.get('nodes')
                    if nodes and len(nodes) == 2:
                        n1, n2 = int(nodes[0]), int(nodes[1])
                        if self.sim.add_bond(n1, n2):
                            self.memory.add_event(f"Added bond {n1}-{n2}")
                # ... (handle other actions like remove_bond, add_node, etc.)
            except Exception as e:
                logging.error(f"Error applying suggestion {action}: {e}")
