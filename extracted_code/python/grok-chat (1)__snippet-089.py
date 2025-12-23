class RelationalLens:
    def __init__(self):
        self.graph = {}  # {entity: {rel: target}}

    def update_relation(self, entity1, relation, entity2):
        if entity1 not in self.graph:
            self.graph[entity1] = {}
        self.graph[entity1][relation] = entity2
        # Bidirectional (BCM sim): Reverse
        if entity2 not in self.graph:
            self.graph[entity2] = {}
        self.graph[entity2][f"inv_{relation}"] = entity1

    def query(self, entity, relation):
        return self.graph.get(entity, {}).get(relation, "Unknown")

