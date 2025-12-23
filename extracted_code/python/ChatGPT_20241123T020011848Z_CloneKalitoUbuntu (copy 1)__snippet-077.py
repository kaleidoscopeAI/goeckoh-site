import json
import random


class Node:
    def __init__(self, node_id, dna, resources):
        self.node_id = node_id
        self.dna = dna
        self.resources = resources
        self.logs = []
        self.synapses = {}

    def replicate(self, threshold, node_id_counter):
        if len(self.resources) >= threshold:
            child_dna = self.dna + f"_child{node_id_counter}"
            new_node = Node(node_id_counter, child_dna, resources={})
            self.logs.append(f"Node {self.node_id} replicated into Node {node_id_counter}.")
            return new_node
        return None

    def communicate(self, other_node):
        shared_data = {"shared_resources": list(self.resources.keys())}
        other_node.receive_data(shared_data)
        self.logs.append(f"Shared data with Node {other_node.node_id}.")

    def receive_data(self, data):
        self.resources.update(data)
        self.logs.append(f"Received data: {data}")

    def learn(self, input_data):
        self.resources.update(input_data)
        self.logs.append(f"Learned new data: {input_data}")

    def store_synapse(self, related_node_id, annotations):
        self.synapses[related_node_id] = annotations
        self.logs.append(f"Stored synapse for Node {related_node_id} with annotations: {annotations}")

    def to_json(self):
        return json.dumps({
            "node_id": self.node_id,
            "dna": self.dna,
            "resources": self.resources,
            "logs": self.logs,
            "synapses": self.synapses
        }, indent=2)

