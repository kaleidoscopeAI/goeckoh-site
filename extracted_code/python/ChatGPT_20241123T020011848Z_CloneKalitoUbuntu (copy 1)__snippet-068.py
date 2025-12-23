from mirrored_network_module import MirroredNetwork
from resource_sharing_module import ResourceSharing
from thought_hub_module import ThoughtHub
from synaptic_overlay_module import SynapticOverlay

# Initialize modules
network = MirroredNetwork()
resource_sharing = ResourceSharing()
thought_hub = ThoughtHub()
synaptic_overlay = SynapticOverlay()

# Populate the network
for i in range(5):
    network.add_node(f"Node_{i}")
    if i > 0:
        network.add_edge(f"Node_{i-1}", f"Node_{i}")

# Share resources and build thought hub
resource_sharing.share_resource("Image_001", {"type": "image", "description": "Hotdog"}, "Node_1")
thought_hub.add_synapse("Node_1", "Hotdog: A food or a dog that is hot.")

# Add synaptic overlay
synaptic_overlay.add_connection("Node_1", "Node_2", "Shares 'Hotdog' context")

# Visualize network and overlay
network.visualize()
synaptic_overlay.visualize_overlay()

# Query examples
print("Query Thought Hub:", thought_hub.query_synapse("Hotdog"))
print("Query Resource Pool:", resource_sharing.query_resource("Image_001"))

