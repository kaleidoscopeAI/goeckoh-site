# __init__ same, n_nodes reduced
async def run_organic_cycle(self, sensor_input=None, web_input=None):
    if not allow_control(np.array([n.awareness for n in self.cube.nodes])):
        logging.warning("Control denied by firewall")
        return "Control denied"

    self.cube.iterate()
    supernodes = self.cube.cluster_supernodes()
    for sn in supernodes:
        sn.reflect(self.cube.transformer)
        reflection = self.cognitive_machine.llm_reflect(sn)
        logging.info(f"Supernode reflection: {reflection}")

    # Device optimization
    S = np.array([n.awareness for n in self.cube.nodes])
    u = optimize_hardware(S, np.array([1.0] * len(S)), np.array([0.5, 0.5, 0.5]))
    apply_device_controls(u)

    # Metrics update (same)
    # ...

    # Viz update
    self.visualizer.update_dashboard(self)

    # Input processing, reflections (same)
    # ...

    return reflection

