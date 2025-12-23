def __init__(self, vector_size: int = 100, window: int = 5, min_count: int = 1, subsampling_threshold: float = 1e-3, negative_samples: int = 5):
    self.vector_size = vector_size
    self.window = window
    self.min_count = min_count
    self.subsampling_threshold = subsampling_threshold
    self.negative_samples = negative_samples
    self.corpus = []
    self.vocabulary = set()
    self.word_counts = defaultdict(int)
    self.word_vectors = {}  # Word embeddings (target words)
    self.context_vectors = {} # Context word embeddings

def train(self, sentences: List[List[str]]):
    """Trains the word embedding model on a list of sentences."""
    self.corpus = sentences
    self._build_vocabulary()
    self._initialize_vectors()
    self._train_model()

def update(self, sentences: List[List[str]]):
    """Updates the model with new sentences, adding to the vocabulary and retraining."""
    self.corpus.extend(sentences)
    self._build_vocabulary()
    self._train_model()

def _build_vocabulary(self):
    """Builds vocabulary and word counts from the corpus."""
    self.vocabulary = set()
    self.word_counts = defaultdict(int)
    for sentence in self.corpus:
        for word in sentence:
            self.word_counts[word] += 1

    # Subsampling of frequent words
    if self.subsampling_threshold > 0:
        self.vocabulary = {
            word for word in self.word_counts
            if self.word_counts[word] >= self.min_count and
            (self.word_counts[word] / len(self.corpus)) < self.subsampling_threshold or
            random.random() < (1 - np.sqrt(self.subsampling_threshold / (self.word_counts[word] / len(self.corpus))))
        }
    else:
        self.vocabulary = {word for word in self.word_counts if self.word_counts[word] >= self.min_count}

def _initialize_vectors(self):
    """Initializes word vectors randomly."""
    for word in self.vocabulary:
        self.word_vectors[word] = np.random.uniform(-0.5, 0.5, self.vector_size)
        self.context_vectors[word] = np.random.uniform(-0.5, 0.5, self.vector_size)

def _train_model(self):
  """Trains the word embedding model using a simplified negative sampling approach."""
  for sentence in self.corpus:
      for i, target_word in enumerate(sentence):
          if target_word not in self.vocabulary:
              continue

          # Get context words within the window
          context_words = sentence[max(0, i - self.window): i] + sentence[i + 1: min(len(sentence), i + self.window + 1)]
          for context_word in context_words:
              if context_word not in self.vocabulary:
                  continue

              # Negative sampling
              negative_samples = self._get_negative_samples(context_word)

              # Update vectors
              self._update_vectors(target_word, context_word, negative_samples)

def _get_negative_samples(self, context_word: str) -> List[str]:
    """Samples negative words for a given context word."""
    not_negative_words = set(context_word)
    negative_samples = []
    while len(negative_samples) < self.negative_samples:
        sample = random.choice(list(self.vocabulary - not_negative_words))
        if sample != context_word:
            negative_samples.append(sample)
    return negative_samples

def _update_vectors(self, target_word: str, context_vector: np.ndarray, label: int, learning_rate: float):
    """Updates word vectors based on positive and negative samples."""
    try:
        context_vector = self.context_vectors[context_word]
        target_vector = self.word_vectors[target_word]
        score = np.dot(target_vector, context_vector)
        prob = 1 / (1 + np.exp(-score))
    except KeyError:
        return  # Do not train for unknown word vectors

    # Calculate the gradient
    gradient = learning_rate * (label - prob) * context_vector

    # Update the word vectors
    self.word_vectors[target_word] += gradient
    context_vector -= learning_rate * (label - prob) * self.word_vectors[target_word]  # Update context vector

def get_vector(self, word: str) -> Optional[np.ndarray]:
    """Returns the word vector for a given word."""
    return self.word_vectors.get(word)

