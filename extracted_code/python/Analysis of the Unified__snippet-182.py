Resources: The Environment starts with initial_resources (default 1000.0) and tracks the resource_history in a deque with a maximum length of 1000.

Node Life Cycle:

    Replication: A node calls replicate() if it meets an internal criteria (implicitly, an energy surplus).

    Removal: A node is explicitly removed from the environment if its energy <= 0.

    Data Flow: The environment is responsible for calling _provide_resources(node) and node.process_input(data) at each simulate_step.

