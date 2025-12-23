# Trigger crystallization when CIAE threshold exceeded
if doc.ciae_score > 0.75:
    crystal_id = self.form_conceptual_crystal(
        doc.embedding, doc.content, doc.url
    )
