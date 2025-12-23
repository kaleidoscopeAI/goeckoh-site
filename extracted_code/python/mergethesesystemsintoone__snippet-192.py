def process(self, data_wrapper: DataWrapper) -> Dict:
    """Processes text data using TF-IDF vectorization."""
    text = data_wrapper.get_data()

    # Fit and transform the text data using the vectorizer
    tfidf_matrix = self.vectorizer.fit_transform([text])

    # Convert to a dense array
    return {"tfidf_vector": tfidf_matrix.toarray()}
