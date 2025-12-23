def __init__(self):
    self.embeddings = []
    self.images = []
    self.metadata = []

def add_image(self, image_path, node_state):
    embedding = embed_image(image_path)
    self.embeddings.append(embedding)
    self.images.append(image_path)
    self.metadata.append({"node_state": node_state, "timestamp": time.time()})

def query_similar(self, query_embedding, top_k=5):
    # Use FAISS or sklearn NearestNeighbors to find similar embeddings
    pass
