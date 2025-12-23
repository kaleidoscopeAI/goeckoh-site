        Removal: A node is explicitly removed from the environment if its energy <= 0.

        Data Flow: The environment is responsible for calling _provide_resources(node) and node.process_input(data) at each simulate_step.

