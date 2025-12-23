def __init__(self, node_id, dna=None):
    self.node_id = node_id
    self.dna = dna if dna is not None else np.random.rand(5)
    self.knowledge_base = {}  # {key: {"description": str, "images": [paths]}}
    self.unique_sources = set()  # Track sources for uniqueness
    self.resources = {"memory": 0.5, "energy": 1.0}

def learn(self, key, description, images, source):
    """Learn only if the source and content are unique."""
    if source in self.unique_sources:
        print(f"Node {self.node_id}: Skipping duplicate source - {source}")
        return

    # Check for unique description
    if any(cosine_similarity([desc], [description])[0][0] > 0.9 for desc in self.knowledge_base.values()):
        print(f"Node {self.node_id}: Skipping redundant knowledge for key - {key}")
        return

    # Check for unique images
    unique_images = []
    for img_path in images:
        img_hash = imagehash.phash(Image.open(img_path))
        if all(img_hash != imagehash.phash(Image.open(existing)) for existing in unique_images):
            unique_images.append(img_path)

    # Store unique knowledge
    self.knowledge_base[key] = {"description": description, "images": unique_images}
    self.unique_sources.add(source)
    print(f"Node {self.node_id}: Learned - {key}")

def share_knowledge(self, other_node):
    """Share unique knowledge with another node."""
    for key, content in self.knowledge_base.items():
        if key not in other_node.knowledge_base:
            other_node.learn(key, content["description"], content["images"], source=f"Node {self.node_id}")

