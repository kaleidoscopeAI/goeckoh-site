def process(self, data_wrapper: DataWrapper) -> np.ndarray:
  """Processes text data using TF-IDF vectorization."""
  text = data_wrapper.get_data()

  # Fit and transform the text data using the vectorizer
  tfidf_matrix = self.vectorizer.fit_transform([text])

  # Convert to a dense array
  return tfidf_matrix.toarray()

def update_vectorizer(self, new_texts: List[str]):
  """Updates the TF-IDF vectorizer with new text data."""
  self.vectorizer.fit(new_texts)
