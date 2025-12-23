import json
import os

class NodeBackupRecovery:
    def __init__(self, backup_dir="backups"):
        self.backup_dir = backup_dir
        os.makedirs(self.backup_dir, exist_ok=True)

    def backup_node(self, node):
        """Backup node state to a file."""
        backup_file = os.path.join(self.backup_dir, f"{node.id}.json")
        with open(backup_file, "w") as f:
            json.dump(node.status(), f)

    def restore_node(self, node_id):
        """Restore node state from a file."""
        backup_file = os.path.join(self.backup_dir, f"{node_id}.json")
        if os.path.exists(backup_file):
            with open(backup_file, "r") as f:
                return json.load(f)
        return None

