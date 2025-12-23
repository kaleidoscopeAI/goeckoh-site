# Project document embeddings to E8 space
def project_document_to_e8(doc: CrawledDocument):
    # Use existing E8Lattice
    projected = e8_lattice.project_to_8d(doc.embedding[:3])
    mirrored = e8_lattice.mirror_state(projected)
    return mirrored
