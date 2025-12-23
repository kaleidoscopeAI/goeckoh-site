class CNodeManager:
    @staticmethod
    def create_c_node():
        node = lib.create_node()
        logging.info("C-level node created.")
        return node

    @staticmethod
    def perform_engrained_action(node):
        lib.engrained_behavior(node)
        if node.contents.memory_used >= 10:
            logging.warning("C-level node memory full. Clearing memory.")
            lib.clear_memory(node)

