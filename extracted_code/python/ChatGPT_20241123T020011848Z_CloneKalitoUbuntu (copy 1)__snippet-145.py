class Environment:
    def __init__(self):
        self.resources = 100  # Example resource limit

    def provide_resources(self, node):
        if self.resources > 0:
            self.resources -= 10  # Allocate a fixed amount per node
            return 10  # Return resource amount provided
        return 0

    def adjust_environment(self, feedback):
        # Adjusts resources based on node activity feedback
        if feedback == "success":
            self.resources += 5  # Replenish resources with positive feedback


